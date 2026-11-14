#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/int32.hpp"

class JoyHandler : public rclcpp::Node
{
public:
  JoyHandler()
  : Node("joy_handler")
  {
    sub_joy_ = this->create_subscription<sensor_msgs::msg::Joy>(
      "/joy", 10,
      std::bind(&JoyHandler::joy_callback, this, std::placeholders::_1));

    // --- ОСИ (6 штук) ---
    pub_host_speed_  = this->create_publisher<std_msgs::msg::Float32>("/host/speed", 10);
    pub_host_dir_  = this->create_publisher<std_msgs::msg::Float32>("/host/dir", 10);
    pub_left_trigger_  = this->create_publisher<std_msgs::msg::Float32>("/gamepad/left_trigger", 10);
    pub_right_stick_x_ = this->create_publisher<std_msgs::msg::Float32>("/gamepad/right_stick_x", 10);
    pub_right_stick_y_ = this->create_publisher<std_msgs::msg::Float32>("/gamepad/right_stick_y", 10);
    pub_right_trigger_ = this->create_publisher<std_msgs::msg::Float32>("/gamepad/right_trigger", 10);

    pub_host_gear_     = this->create_publisher<std_msgs::msg::Int32>("/host/gear", 10);
    // --- КНОПКИ (10 штук) ---
    pub_btn_x_         = this->create_publisher<std_msgs::msg::Bool>("/gamepad/btn_x", 10);
    pub_btn_y_         = this->create_publisher<std_msgs::msg::Bool>("/gamepad/btn_y", 10);
    pub_btn_lb_        = this->create_publisher<std_msgs::msg::Bool>("/gamepad/btn_lb", 10);
    pub_btn_rb_        = this->create_publisher<std_msgs::msg::Bool>("/gamepad/btn_rb", 10);
    pub_btn_back_      = this->create_publisher<std_msgs::msg::Bool>("/gamepad/btn_back", 10);
    pub_btn_start_     = this->create_publisher<std_msgs::msg::Bool>("/gamepad/btn_start", 10);
    pub_btn_guide_     = this->create_publisher<std_msgs::msg::Bool>("/gamepad/btn_guide", 10);
    pub_btn_l_stick_   = this->create_publisher<std_msgs::msg::Bool>("/gamepad/btn_left_stick", 10);
    pub_btn_r_stick_   = this->create_publisher<std_msgs::msg::Bool>("/gamepad/btn_right_stick", 10);
  }

private:
  short gear = 0;
  bool gearchanged = 0, gear_cooldown_passed = 0;
  rclcpp::Time last_gear_time_{0, 0, RCL_ROS_TIME};
  void publish_araw(const float data,
                    const rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr & pub)
  {
    std_msgs::msg::Float32 out;
    out.data = data;
    pub->publish(out);
  }
  void publish_iraw(const int data,
                    const rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr & pub)
  {
    std_msgs::msg::Int32 out;
    out.data = data;
    pub->publish(out);
  }
  void publish_axis(const sensor_msgs::msg::Joy::SharedPtr & msg, size_t idx,
                    const rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr & pub)
  {
    if (idx < msg->axes.size()) {
      std_msgs::msg::Float32 out;
      out.data = msg->axes[idx];
      pub->publish(out);
    }
  }

  void publish_button(const sensor_msgs::msg::Joy::SharedPtr & msg, size_t idx,
                      const rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr & pub)
  {
    if (idx < msg->buttons.size()) {
      std_msgs::msg::Bool out;
      out.data = (msg->buttons[idx] != 0);
      pub->publish(out);
    }
  }

  void joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg)
  {
    auto now = this->now();
    gear_cooldown_passed = (now - last_gear_time_).seconds() >= 0.3;
    publish_axis(msg, 0, pub_host_speed_);
    publish_axis(msg, 1, pub_host_dir_);
    publish_axis(msg, 2, pub_right_stick_y_);
    publish_axis(msg, 3, pub_right_stick_x_);
    publish_axis(msg, 4, pub_left_trigger_);
    publish_axis(msg, 5, pub_right_trigger_);

    
    if (msg->buttons[0] == 1 && !gearchanged && gear_cooldown_passed){
      if (gear < 3) gear++;
      gearchanged = 1;
      last_gear_time_ = now;
      publish_iraw(gear, pub_host_gear_);
    }
    else if (msg->buttons[1] == 1 && !gearchanged && gear_cooldown_passed){
      if (gear > 0) gear--;
      gearchanged = 1;
      last_gear_time_ = now;
      publish_iraw(gear, pub_host_gear_);
    }
    else if (msg->buttons[0] == 0 && msg->buttons[1] == 0 && gearchanged) gearchanged = 0;

    publish_button(msg, 2, pub_btn_x_);
    publish_button(msg, 3, pub_btn_y_);
    publish_button(msg, 4, pub_btn_lb_);
    publish_button(msg, 5, pub_btn_rb_);
    publish_button(msg, 6, pub_btn_back_);
    publish_button(msg, 7, pub_btn_start_);
    publish_button(msg, 8, pub_btn_guide_);
    publish_button(msg, 9, pub_btn_l_stick_);
    publish_button(msg, 10, pub_btn_r_stick_);
  }

  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr sub_joy_;

  // Float32 vars
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pub_host_dir_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pub_host_speed_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pub_left_trigger_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pub_right_stick_x_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pub_right_stick_y_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pub_right_trigger_;

  // Int vars
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr pub_host_gear_;

  // Bool vars
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_btn_x_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_btn_y_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_btn_lb_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_btn_rb_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_btn_back_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_btn_start_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_btn_guide_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_btn_l_stick_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_btn_r_stick_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<JoyHandler>());
  rclcpp::shutdown();
  return 0;
}