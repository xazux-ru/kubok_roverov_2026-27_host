#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/int32.hpp>
#include <opencv2/opencv.hpp>
#ifdef VR_HUD_USE_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace
{
    constexpr int EYE_W = 1080;
    constexpr int EYE_H = 1200;

    inline int hud_x(bool is_left) { return is_left ? 215 : 120; }

    constexpr int SLOT_ROW_W = 810;
    constexpr int SLOT_ROW_H = 50;
    constexpr int SLOT_ROW_TOP_CY = 300;
    constexpr int SLOT_FONT = cv::FONT_HERSHEY_SIMPLEX;
    constexpr double SLOT_SCALE = 0.7;
    constexpr int SLOT_THICK = 2;

    // Offsets for HMD
    inline int slot_row_y(int row)
    {
        const int cy = (row == 0) ? SLOT_ROW_TOP_CY : (EYE_H - SLOT_ROW_TOP_CY);
        return cy - SLOT_ROW_H / 2;
    }
    inline int slot_row_x(bool is_left)
    {
        const int center = EYE_W / 2 + hud_x(is_left) - (hud_x(true) + hud_x(false)) / 2;
        return center - SLOT_ROW_W / 2;
    }
} // namespace

class VrHudNode : public rclcpp::Node
{
public:
    VrHudNode() : Node("vr_hud_node")
    {
        auto video_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();

        image_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
            "/image_raw/compressed", video_qos,
            [this](const sensor_msgs::msg::CompressedImage::SharedPtr msg)
            { image_callback(msg); });

        std::vector<std::string> def_topics = {
            "/gamepad/left_trigger",
            "/rover/gear,/host/gear",
            "/gamepad/left_stick_x",
            "/gamepad/left_stick_y",
            "/gamepad/right_stick_x",
            "/gamepad/right_stick_y"};
        std::vector<std::string> def_labels = {"LT", "G", "LX", "LY", "RX", "RY"};
        std::vector<std::string> def_units = {"", "", "", "", "", ""};

        auto slot_param = [this](const std::string &name, const std::vector<std::string> &def)
        {
            this->declare_parameter<std::vector<std::string>>(name, def);
            std::vector<std::string> v = this->get_parameter(name).as_string_array();
            while (v.size() < 6)
                v.push_back(def[v.size()]);
            return v;
        };

        const auto topics = slot_param("slot_topics", def_topics);
        const auto labels = slot_param("slot_labels", def_labels);
        const auto units = slot_param("slot_units", def_units);

        auto bind_topic = [this](const std::string &topic, std::atomic<float> &target, bool is_int)
        {
            if (is_int)
            {
                slot_subs_.push_back(this->create_subscription<std_msgs::msg::Int32>(
                    topic, 10, [&target](const std_msgs::msg::Int32::SharedPtr msg)
                    { target = static_cast<float>(msg->data); }));
            }
            else
            {
                slot_subs_.push_back(this->create_subscription<std_msgs::msg::Float32>(
                    topic, 10, [&target](const std_msgs::msg::Float32::SharedPtr msg)
                    { target = msg->data; }));
            }
        };

        for (size_t i = 0; i < 6; ++i)
        {
            slots_[i].label = labels[i];
            slots_[i].unit = units[i];
            slots_[i].is_int = (topics[i].find("gear") != std::string::npos);

            const auto comma = topics[i].find(',');
            if (comma == std::string::npos)
            {
                bind_topic(topics[i], slots_[i].val1, slots_[i].is_int);
            }
            else
            {
                slots_[i].has_second = true;
                bind_topic(topics[i].substr(0, comma), slots_[i].val1, slots_[i].is_int);
                bind_topic(topics[i].substr(comma + 1), slots_[i].val2, slots_[i].is_int);
            }
            RCLCPP_INFO(get_logger(), "HUD slot %zu: '%s' <- %s", i, labels[i].c_str(), topics[i].c_str());
        }

        this->declare_parameter<double>("image_zoom", 1.0);
        this->declare_parameter<bool>("use_gpu", true);
        image_zoom_ = clamp_zoom(static_cast<float>(this->get_parameter("image_zoom").as_double()));

        const bool want_gpu = this->get_parameter("use_gpu").as_bool();
#ifdef VR_HUD_USE_CUDA
        if (want_gpu)
        {
            if (cv::cuda::getCudaEnabledDeviceCount() > 0)
            {
                cv::cuda::setDevice(0);
                gpu_sbs_.create(EYE_H, 2 * EYE_W, CV_8UC3);
                gpu_canvas_.create(EYE_H, EYE_W, CV_8UC3);
                pinned_ = cv::cuda::HostMem(EYE_H, 2 * EYE_W, CV_8UC3, cv::cuda::HostMem::PAGE_LOCKED);
                use_gpu_ = true;
            }
            else
            {
                RCLCPP_WARN(get_logger(), "CUDA-устройство не найдено, работаю на CPU");
            }
        }
#else
        if (want_gpu)
        {
            RCLCPP_WARN(get_logger(), "Собрано без VR_HUD_USE_CUDA, работаю на CPU");
        }
#endif
        RCLCPP_INFO(get_logger(), "Режим рендера: %s", use_gpu_ ? "GPU (CUDA)" : "CPU");

        sbs_cpu_.create(EYE_H, 2 * EYE_W, CV_8UC3);
        canvas_cpu_.create(EYE_H, EYE_W, CV_8UC3);
        update_lens_maps(0.0f);
    }

    static float clamp_zoom(float z) { return std::max(0.3f, std::min(z, 2.0f)); }

    float change_zoom(float delta)
    {
        float z = clamp_zoom(image_zoom_.load() + delta);
        image_zoom_ = z;
        return z;
    }

    void update_lens_maps(float ipd_center_shift)
    {
        ipd_shift_ = ipd_center_shift;
        for (int e = 0; e < 2; e++)
        {
            const bool is_left = (e == 0);
            generate_lens_maps(is_left, ipd_center_shift, map_x_[e], map_y_[e]);

            float optical_offset = 45.0f + (is_left ? ipd_center_shift : -ipd_center_shift);
            cv::Point lens_center(EYE_W / 2 + (is_left ? optical_offset : -optical_offset), EYE_H / 2);
            mask_cpu_[e] = cv::Mat::zeros(EYE_H, EYE_W, CV_8UC1);
            cv::circle(mask_cpu_[e], lens_center, 535, cv::Scalar(255), -1, cv::LINE_AA);

#ifdef VR_HUD_USE_CUDA
            if (use_gpu_)
            {
                gpu_map_x_[e].upload(map_x_[e]);
                gpu_map_y_[e].upload(map_y_[e]);
                gpu_mask_[e].upload(mask_cpu_[e]);
            }
#endif
        }
        // Вне круга маски пиксели больше не пишутся, поэтому чистим холст при смене маски
        sbs_cpu_.setTo(cv::Scalar::all(0));
#ifdef VR_HUD_USE_CUDA
        if (use_gpu_)
            gpu_sbs_.setTo(cv::Scalar::all(0));
#endif
        force_redraw_ = true;
    }

    // Рендерит стереопару. Возвращает true, если появилась новая картинка для показа.
    bool render_sbs(cv::Mat &out)
    {
        cv::Mat frame;
        uint64_t seq;
        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            frame = latest_frame_; // без clone: декодированный кадр после записи не меняется
            seq = frame_seq_;
        }

        if (frame.empty())
        {
            if (waiting_shown_ && !force_redraw_)
                return false;
            waiting_img_ = cv::Mat::zeros(EYE_H, 2 * EYE_W, CV_8UC3);
            for (int e = 0; e < 2; e++)
            {
                cv::putText(waiting_img_, "WAITING FOR CAMERA...", cv::Point(e * EYE_W + hud_x(e == 0) + 170, 600),
                            cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 3);
            }
            waiting_shown_ = true;
            force_redraw_ = false;
            out = waiting_img_;
            return true;
        }

        const float zoom = image_zoom_.load();
        const bool new_frame = (seq != last_seq_);
        const bool zoom_changed = (zoom != last_zoom_);

        std::array<std::string, 6> slot_strs;
        for (size_t i = 0; i < 6; ++i)
            slot_strs[i] = slots_[i].format();
        const bool hud_changed = (slot_strs != hud_slot_strs_);

        // Нечего перерисовывать - не тратим время
        if (!new_frame && !zoom_changed && !hud_changed && !force_redraw_)
            return false;

        last_seq_ = seq;
        last_zoom_ = zoom;
        force_redraw_ = false;
        hud_slot_strs_ = slot_strs;

#ifdef VR_HUD_USE_CUDA
        if (use_gpu_)
        {
            render_gpu(frame, new_frame, new_frame || zoom_changed, zoom, hud_changed, out);
            return true;
        }
#endif
        render_cpu(frame, zoom, out);
        return true;
    }

private:
    struct HudSlot
    {
        std::string label, unit;
        bool is_int = false;
        bool has_second = false;
        std::atomic<float> val1{std::nanf("")}; // NaN = данных ещё нет
        std::atomic<float> val2{std::nanf("")};

        std::string format_val(float v) const
        {
            if (std::isnan(v))
                return "--";
            return is_int ? std::to_string(static_cast<int>(std::lround(v))) : cv::format("%.1f", v);
        }

        std::string format() const
        {
            std::string s = label + ": " + format_val(val1.load());
            if (has_second)
                s += "/" + format_val(val2.load());
            if (!unit.empty())
                s += " " + unit;
            return s;
        }
    };

    // Рисует один ряд из 3 слотов (слева / центр / справа) в области SLOT_ROW_W x SLOT_ROW_H
    // с левым верхним углом (ox, oy). Текст центрируется в своей колонке.
    void draw_slot_row(cv::Mat &img, int row, int ox, int oy) const
    {
        const int col_w = SLOT_ROW_W / 3;
        for (int c = 0; c < 3; ++c)
        {
            const std::string &s = hud_slot_strs_[row * 3 + c];
            int base = 0;
            const cv::Size ts = cv::getTextSize(s, SLOT_FONT, SLOT_SCALE, SLOT_THICK, &base);
            const int cx = ox + c * col_w + col_w / 2;
            cv::putText(img, s, cv::Point(cx - ts.width / 2, oy + SLOT_ROW_H / 2 + ts.height / 2),
                        SLOT_FONT, SLOT_SCALE, cv::Scalar(0, 255, 0), SLOT_THICK, cv::LINE_AA);
        }
    }
    struct Placement
    {
        int tw, th; // размер масштабированной картинки
        int x, y;   // позиция левого верхнего угла на холсте глаза
        float scale;
    };
    // Центр картинки ВСЕГДА стоит в одной точке холста глаза (540 +/- 50), независимо от зума.
    static Placement compute_placement(int cols, int rows, float zoom, bool is_left)
    {
        float scale = std::max(static_cast<float>(EYE_W) / cols, static_cast<float>(EYE_H) / rows) * zoom;
        int tw = std::max(1, static_cast<int>(std::lround(cols * scale)));
        int th = std::max(1, static_cast<int>(std::lround(rows * scale)));
        const int texture_shift = -50;
        int cx = EYE_W / 2 + (is_left ? -texture_shift : texture_shift);
        int x = cx - tw / 2;
        int y = EYE_H / 2 - th / 2;
        // Зажим по краям только при зуме >= 1: при зуме < 1 он ломает постоянное
        // стерео-смещение между глазами и вызывает двоение.
        if (zoom >= 1.0f)
        {
            if (tw >= EYE_W)
                x = std::max(EYE_W - tw, std::min(x, 0));
            if (th >= EYE_H)
                y = std::max(EYE_H - th, std::min(y, 0));
        }
        return {tw, th, x, y, scale};
    }
    static bool placement_rects(const Placement &p, cv::Rect &dst, cv::Rect &src)
    {
        dst = cv::Rect(p.x, p.y, p.tw, p.th) & cv::Rect(0, 0, EYE_W, EYE_H);
        if (dst.area() <= 0)
            return false;
        src = cv::Rect(dst.x - p.x, dst.y - p.y, dst.width, dst.height);
        return true;
    }
    // ---------------- CPU-путь ----------------
    void render_cpu(const cv::Mat &frame, float zoom, cv::Mat &out)
    {
        Placement pl[2] = {compute_placement(frame.cols, frame.rows, zoom, true),
                           compute_placement(frame.cols, frame.rows, zoom, false)};
        // Масштабируем ОДИН раз для обоих глаз
        cv::resize(frame, resized_cpu_, cv::Size(pl[0].tw, pl[0].th), 0, 0,
                   pl[0].scale < 1.0f ? cv::INTER_AREA : cv::INTER_LINEAR);
                for (int e = 0; e < 2; e++)
                {
                    const bool is_left = (e == 0);
                    canvas_cpu_.setTo(cv::Scalar::all(0));
                    cv::Rect dst, src;
                    if (placement_rects(pl[e], dst, src))
                    {
                        resized_cpu_(src).copyTo(canvas_cpu_(dst));
                    }
                    for (int r = 0; r < 2; ++r)
                    {
                        draw_slot_row(canvas_cpu_, r, slot_row_x(is_left), slot_row_y(r));
                    }
                    cv::remap(canvas_cpu_, distorted_cpu_, map_x_[e], map_y_[e],
                            cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
                    cv::Mat eye_out = sbs_cpu_(cv::Rect(e * EYE_W, 0, EYE_W, EYE_H));
                    distorted_cpu_.copyTo(eye_out, mask_cpu_[e]);
                }
        out = sbs_cpu_;
    }
#ifdef VR_HUD_USE_CUDA
    void rebuild_hud_overlay()
    {
        for (int r = 0; r < 2; ++r)
        {
            cv::Mat row_img = cv::Mat::zeros(SLOT_ROW_H, SLOT_ROW_W, CV_8UC3);
            draw_slot_row(row_img, r, 0, 0);
            cv::Mat row_gray, row_mask;
            cv::cvtColor(row_img, row_gray, cv::COLOR_BGR2GRAY);
            cv::threshold(row_gray, row_mask, 0, 255, cv::THRESH_BINARY);
            gpu_row_img_[r].upload(row_img);
            gpu_row_mask_[r].upload(row_mask);
        }
    }

    void render_gpu(const cv::Mat &frame, bool new_frame, bool need_resize, float zoom,
                    bool hud_changed, cv::Mat &out)
    {
        if (new_frame)
            gpu_frame_.upload(frame);
        Placement pl[2] = {compute_placement(frame.cols, frame.rows, zoom, true),
                           compute_placement(frame.cols, frame.rows, zoom, false)};
        if (need_resize)
        {
            cv::cuda::resize(gpu_frame_, gpu_resized_, cv::Size(pl[0].tw, pl[0].th), 0, 0,
                             pl[0].scale < 1.0f ? cv::INTER_AREA : cv::INTER_LINEAR);
        }
                if (hud_changed || gpu_row_img_[0].empty())
            rebuild_hud_overlay();
                if (hud_changed || gpu_row_img_[0].empty())
            rebuild_hud_overlay();
        if (hud_changed || gpu_row_img_[0].empty())
            rebuild_hud_overlay();
        for (int e = 0; e < 2; e++)
        {
            const bool is_left = (e == 0);
            gpu_canvas_.setTo(cv::Scalar::all(0));
            cv::Rect dst, src;
            if (placement_rects(pl[e], dst, src))
            {
                cv::cuda::GpuMat roi = gpu_canvas_(dst);
                gpu_resized_(src).copyTo(roi);
            }
            for (int r = 0; r < 2; ++r)
            {
                cv::cuda::GpuMat row_roi =
                    gpu_canvas_(cv::Rect(slot_row_x(is_left), slot_row_y(r), SLOT_ROW_W, SLOT_ROW_H));
                gpu_row_img_[r].copyTo(row_roi, gpu_row_mask_[r]);
            }
            cv::cuda::remap(gpu_canvas_, gpu_distorted_, gpu_map_x_[e], gpu_map_y_[e],
                            cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
            cv::cuda::GpuMat eye_out = gpu_sbs_(cv::Rect(e * EYE_W, 0, EYE_W, EYE_H));
            gpu_distorted_.copyTo(eye_out, gpu_mask_[e]);
        }
        gpu_sbs_.download(pinned_);
        out = pinned_.createMatHeader();
    }
#endif
    void image_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
    {
        if (msg->data.empty())
            return;
        cv::Mat decoded = cv::imdecode(msg->data, cv::IMREAD_COLOR);
        if (decoded.empty())
            return;
        std::lock_guard<std::mutex> lock(frame_mutex_);
        latest_frame_ = decoded;
        ++frame_seq_;
    }
    void generate_lens_maps(bool is_left, float ipd_center_shift, cv::Mat &map_x, cv::Mat &map_y)
    {
        const int w = EYE_W;
        const int h = EYE_H;
        map_x.create(h, w, CV_32FC1);
        map_y.create(h, w, CV_32FC1);
        const float k1 = 0.26f;
        const float k2 = 0.16f;
        float optical_offset = 45.0f + (is_left ? ipd_center_shift : -ipd_center_shift);
        const float cx = (w / 2.0f) + (is_left ? optical_offset : -optical_offset);
        const float cy = h / 2.0f;
        cv::parallel_for_(cv::Range(0, h), [&](const cv::Range &range)
                          {
for (int y = range.start; y < range.end; y++) {
float* px = map_x.ptr<float>(y);
float* py = map_y.ptr<float>(y);
for (int x = 0; x < w; x++) {
float nx = (x - cx) / (w / 2.0f);
float ny = (y - cy) / (h / 2.0f);
float r2 = nx * nx + ny * ny;
float distortion = 1.0f + k1 * r2 + k2 * r2 * r2;
px[x] = nx * distortion * (w / 2.0f) + cx;
py[x] = ny * distortion * (h / 2.0f) + cy;
}
} });
    }
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_sub_;
    std::vector<rclcpp::SubscriptionBase::SharedPtr> slot_subs_;
    std::array<HudSlot, 6> slots_; // значения пишет ROS-поток, читает главный
    // Данные между ROS-потоком и главным потоком
    cv::Mat latest_frame_;
    uint64_t frame_seq_ = 0;
    std::mutex frame_mutex_;
    std::atomic<float> image_zoom_{1.0f};
    // Всё ниже используется только главным потоком
    bool use_gpu_ = false;
    bool force_redraw_ = true;
    bool waiting_shown_ = false;
    uint64_t last_seq_ = 0;
    float last_zoom_ = -1.0f;
    float ipd_shift_ = 0.0f;
    std::array<std::string, 6> hud_slot_strs_;
    cv::Mat map_x_[2], map_y_[2], mask_cpu_[2];
    cv::Mat sbs_cpu_, canvas_cpu_, resized_cpu_, distorted_cpu_, waiting_img_;
#ifdef VR_HUD_USE_CUDA
    cv::cuda::GpuMat gpu_frame_, gpu_resized_, gpu_canvas_, gpu_distorted_, gpu_sbs_;
    cv::cuda::GpuMat gpu_map_x_[2], gpu_map_y_[2], gpu_mask_[2];
    cv::cuda::GpuMat gpu_row_img_[2], gpu_row_mask_[2];
    cv::cuda::HostMem pinned_;
#endif
};
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VrHudNode>();
    std::thread ros_thread([&]() { rclcpp::spin(node); });
    std::string window_name = "VR HUD Extended Window";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::setWindowProperty(window_name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_NORMAL);
    float ipd_center_shift = 0.0f;
    RCLCPP_INFO(node->get_logger(), "VR HUD Oculus-Mask style started. Use 'A'/'D' for IPD, 'W'/'S' (or +/-) for image zoom!");
    cv::Mat frame_out;
    while (rclcpp::ok())
    {
        if (node->render_sbs(frame_out))
        {
            cv::imshow(window_name, frame_out);
        }
        int key = cv::waitKey(1);
        if (key == 27)
        { // ESC
            break;
        }
        else if (key == 121 || key == 201)
        { // F11
            double prop = cv::getWindowProperty(window_name, cv::WND_PROP_FULLSCREEN);
            if (prop == cv::WINDOW_FULLSCREEN)
            {
                cv::setWindowProperty(window_name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_NORMAL);
            }
            else
            {
                cv::setWindowProperty(window_name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
            }
        }
        else if (key == 's' || key == 'S' || key == '-')
        {
            RCLCPP_INFO(node->get_logger(), "Image zoom: %.2f", node->change_zoom(-0.02f));
        }
        else if (key == 'w' || key == 'W' || key == '+' || key == '=')
        {
            RCLCPP_INFO(node->get_logger(), "Image zoom: %.2f", node->change_zoom(+0.02f));
        }
        else if (key == 'a' || key == 'A')
        {
            ipd_center_shift -= 1.0f;
            node->update_lens_maps(ipd_center_shift);
            RCLCPP_INFO(node->get_logger(), "Optical Shift: %.1f", ipd_center_shift);
        }
        else if (key == 'd' || key == 'D')
        {
            ipd_center_shift += 1.0f;
            node->update_lens_maps(ipd_center_shift);
            RCLCPP_INFO(node->get_logger(), "Optical Shift: %.1f", ipd_center_shift);
        }
    }
    cv::destroyAllWindows();
    rclcpp::shutdown();
    ros_thread.join();
    return 0;
}