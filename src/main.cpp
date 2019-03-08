#include <uWS/uWS.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "json.hpp"
#include "spline.h"
#include <memory>

// for convenience
using nlohmann::json;
using std::string;
using std::vector;

constexpr static double LANE_SIZE = 4.0;
constexpr static double HALF_LANE_SIZE = LANE_SIZE / 2.0;
constexpr static double SPEED_LIMIT = 45; // mph
constexpr static double SPEED_INCREMENT = .224; // m/s^2
constexpr static double FRONT_MARGIN = 20; // meters
constexpr static double BACK_MARGIN = 20; // meters
constexpr static int32_t LEFTMOST_LANE = 0;
constexpr static int32_t RIGHTMOST_LANE = 2;
constexpr static int32_t NUM_LANES = RIGHTMOST_LANE + 1;

class vehicle;

enum vehicle_state
{
    ST_READY,
    ST_PREPARELANECHANGERIGHT,
    ST_PREPARELANECHANGELEFT,
    ST_LANECHANGERIGHT,
    ST_LANECHANGELEFT,
    ST_LANEKEEP
};

class vehicle
{
public:
    double s = 0;
    int lane = 1;
    double ref_vel = 0;
    int prev_size = 0;
    vehicle_state state = ST_LANEKEEP;

    vehicle() = default;
    virtual ~vehicle() = default;

    std::string get_state_name(vehicle_state s)
    {
        switch(s)
        {
            case ST_READY:
                return "READY";
            case ST_PREPARELANECHANGERIGHT:
                return "PREPARELANECHANGERIGHT";
            case ST_PREPARELANECHANGELEFT:
                return "PREPARELANECHANGELEFT";
            case ST_LANECHANGERIGHT:
                return "LANECHANGERIGHT";
            case ST_LANECHANGELEFT:
                return "ST_LANECHANGELEFT";
            case ST_LANEKEEP:
                return "LANEKEEP";
            default:
                return "UNKNOWN";
        }
    }

    inline int get_lane(double car_d)
    {
        for (int i = LEFTMOST_LANE; i <= RIGHTMOST_LANE; ++i)
        {
            if (car_d > LANE_SIZE * i and car_d < LANE_SIZE * i + LANE_SIZE)
            {
                return i;
            }
        }
        return -1;
    }

    inline double lane_leftmost_d()
    {
        return LANE_SIZE * lane;
    }

    inline double lane_rightmost_d()
    {
        return LANE_SIZE * lane + LANE_SIZE;
    }

    inline double lane_center_d()
    {
        return LANE_SIZE * lane + HALF_LANE_SIZE;
    }

    inline bool is_on_vehicle_lane(double car_d)
    {
        return car_d < lane_rightmost_d() and car_d > lane_leftmost_d();
    }

    inline bool is_on_vehicle_left_lane(double car_d)
    {
        int car_lane = get_lane(car_d);
        return car_lane != -1 and car_lane < lane;
    }

    inline bool is_on_vehicle_right_lane(double car_d)
    {
        int car_lane = get_lane(car_d);
        return car_lane != -1 and car_lane > lane;
    }

    inline bool is_in_front(double car_s)
    {
        return car_s > s;
    }

    inline bool is_in_back(double car_s)
    {
        return car_s < s;
    }

    inline bool is_safe_front_margin(double car_s)
    {
        return (car_s - s) >= FRONT_MARGIN;
    }

    inline bool is_safe_back_margin(double car_s)
    {
        return (s - car_s) >= BACK_MARGIN;
    }

    inline bool is_safe_changelane_margin(double car_s)
    {
        if (is_in_front(car_s))
        {
            return is_safe_front_margin(car_s);
        }
        return is_safe_back_margin(car_s);
    }

    inline bool is_leftmost_lane()
    {
        return lane == LEFTMOST_LANE;
    }

    inline bool is_rightmost_lane()
    {
        return lane == RIGHTMOST_LANE;
    }

    double lane_transition_cost(double target_speed, int current_lane, int final_lane,
                                const std::vector<double> &lane_speeds)
    {
        double current_speed = lane_speeds[current_lane];
        double final_speed = lane_speeds[final_lane];
        double cost = (2.0*target_speed - current_speed - final_speed)/target_speed;
        return cost;
    }

    void update(const std::vector<std::vector<double>>& sensor_fusion)
    {
        bool too_close = false;
        bool safe_change_left = true;
        bool safe_change_right = true;
        bool prepare_lane_change = false;

        std::vector<double> lane_speeds(NUM_LANES, SPEED_LIMIT);
        std::vector<double> lane_speeds_behind(NUM_LANES, 0);

        std::vector<int> closest_front_car(NUM_LANES, -1);
        std::vector<int> closest_behind_car(NUM_LANES, -1);
        for (int i = 0; i < sensor_fusion.size(); ++i)
        {
            double car_vx = sensor_fusion[i][3];
            double car_vy = sensor_fusion[i][4];
            double car_speed = sqrt(car_vx*car_vx + car_vy*car_vy);
            double car_s = sensor_fusion[i][5];
            double car_d = sensor_fusion[i][6];
            car_s += ((double) prev_size * .02 * car_speed);
            int car_lane = get_lane(car_d);

            //std::cout << car_lane << " " << car_d  << " " << car_s << '\n';

            /// closest car on the front
            if (car_lane != -1 and is_in_front(car_s))
            {
                int best_idx = closest_front_car[car_lane];
                if (best_idx == -1 or car_s - s < sensor_fusion[best_idx][5] - s)
                {
                    closest_front_car[car_lane] = i;
                }
            }

            /// closest car on the back
            if (car_lane != -1 and is_in_back(car_s))
            {
                int best_idx = closest_behind_car[car_lane];
                if (best_idx == -1 or s - car_s < s - sensor_fusion[best_idx][5])
                {
                    closest_behind_car[car_lane] = i;
                }
            }

            /// car is on vehicle lane
            if (is_on_vehicle_lane(car_d))
            {
                if (is_in_front(car_s) && not is_safe_front_margin(car_s))
                {
                    too_close = true;
                }
            }

            /// car is on vehicle left lane
            if (is_on_vehicle_left_lane(car_d))
            {
                if (not is_safe_changelane_margin(car_s))
                {
                    safe_change_left = false;
                }
            }

            /// car is on vehicle right lane
            if (is_on_vehicle_right_lane(car_d))
            {
                if (not is_safe_changelane_margin(car_s))
                {
                    safe_change_right = false;
                }
            }
        }

        for (int i = 0; i < NUM_LANES; ++i)
        {
            /// lane speeds front
            if (closest_front_car[i] != -1)
            {
                int best_idx = closest_front_car[i];
                double car_vx = sensor_fusion[best_idx][3];
                double car_vy = sensor_fusion[best_idx][4];
                double car_speed = sqrt(car_vx * car_vx + car_vy * car_vy);
                double car_s = sensor_fusion[best_idx][5];

                if (car_s - s < 30)
                {
                    lane_speeds[i] = car_speed;
                }
                else
                {
                    lane_speeds[i] = SPEED_LIMIT;
                }
            }

            /// lane speeds behind front
            if (closest_behind_car[i] != -1)
            {
                int best_idx = closest_behind_car[i];
                double car_vx = sensor_fusion[best_idx][3];
                double car_vy = sensor_fusion[best_idx][4];
                double car_speed = sqrt(car_vx * car_vx + car_vy * car_vy);
                lane_speeds_behind[i] = car_speed;
                double car_s = sensor_fusion[best_idx][5];

                if (s - car_s < 30)
                {
                    lane_speeds_behind[i] = car_speed;
                }
                else
                {
                    lane_speeds_behind[i] = 0;
                }
            }
        }

        if (state == ST_LANEKEEP)
        {
            /// calculate costs
            std::vector<vehicle_state> successor_states;
            successor_states.push_back(vehicle_state::ST_LANEKEEP);

            if (safe_change_left and lane - 1 >= LEFTMOST_LANE)
            {
                successor_states.push_back(vehicle_state::ST_PREPARELANECHANGELEFT);
            }

            if (safe_change_right and lane + 1 <= RIGHTMOST_LANE)
            {
                successor_states.push_back(vehicle_state::ST_PREPARELANECHANGERIGHT);
            }

            std::vector<double> costs;
            for (auto next_state: successor_states)
            {
                double cost = 0;
                switch (next_state)
                {
                    case vehicle_state::ST_LANEKEEP:
                    {
                        cost = lane_transition_cost(SPEED_LIMIT, lane, lane, lane_speeds);
                        costs.push_back(cost);
                        break;
                    }
                    case vehicle_state::ST_PREPARELANECHANGELEFT:
                    {
                        cost = lane_transition_cost(SPEED_LIMIT, lane, lane - 1, lane_speeds);
                        costs.push_back(cost);
                        break;
                    }
                    case vehicle_state::ST_PREPARELANECHANGERIGHT:
                    {
                        cost = lane_transition_cost(SPEED_LIMIT, lane, lane + 1, lane_speeds);
                        costs.push_back(cost);
                        break;
                    }
                    default:
                    {
                        // do nothing
                        break;
                    }
                }
                std::cout << get_state_name(next_state) << " = " << cost << std::endl;
            }

            auto best_cost = std::min_element(begin(costs), end(costs));
            auto best_idx = std::distance(begin(costs), best_cost);
            state = successor_states[best_idx];
        }
        else if (state == ST_PREPARELANECHANGELEFT)
        {
            if (ref_vel - lane_speeds_behind[lane - 1] > 15)
            {
                if (safe_change_left and ref_vel < 30)
                {
                    state = ST_LANECHANGELEFT;
                }
                else
                {
                    prepare_lane_change = true;
                }
            }
        }
        else if (state == ST_PREPARELANECHANGERIGHT)
        {
            if (ref_vel - lane_speeds_behind[lane + 1] > 15)
            {
                if (safe_change_right and ref_vel < 30)
                {
                    state = ST_LANECHANGERIGHT;
                }
                else
                {
                    prepare_lane_change = true;
                }
            }

        }
        else if (state == ST_LANECHANGELEFT)
        {
            lane = lane - 1;
            state = ST_LANEKEEP;
        }
        else if (state == ST_LANECHANGERIGHT)
        {
            lane = lane + 1;
            state = ST_LANEKEEP;
        }

        std::cout << "ref_vel=" << ref_vel << " " << get_state_name(state) << " lane_speeds = " << lane_speeds[0] << " " << lane_speeds[1] << " " << lane_speeds[2] << " "
                  << " lane_speeds_behind = " << lane_speeds_behind[0] << " " << lane_speeds_behind[1] << " " << lane_speeds_behind[2] << " "
                  << ", safe_change_left = " << safe_change_left << ", safe_change_right = " << safe_change_right << "\n";

        if (too_close or prepare_lane_change)
        {
            ref_vel -= SPEED_INCREMENT;
        }
        else if (ref_vel < SPEED_LIMIT)
        {
            ref_vel += SPEED_INCREMENT;
        }
    }
};



int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    std::istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }

  /// start lane in 1
  int lane = 1;

  /// reference velocity in mph
  double ref_vel = 0;

  vehicle vehicle;

  h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,
               &map_waypoints_dx,&map_waypoints_dy, &vehicle]
              (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
               uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
          // Main car's localization Data
          double car_x = j[1]["x"];
          double car_y = j[1]["y"];
          double car_s = j[1]["s"];
          double car_d = j[1]["d"];
          double car_yaw = j[1]["yaw"];
          double car_speed = j[1]["speed"];

          // Previous path data given to the Planner
          auto previous_path_x = j[1]["previous_path_x"];
          auto previous_path_y = j[1]["previous_path_y"];
          // Previous path's end s and d values 
          double end_path_s = j[1]["end_path_s"];
          double end_path_d = j[1]["end_path_d"];

          // Sensor Fusion Data, a list of all other cars on the same side 
          //   of the road.
          auto sensor_fusion = j[1]["sensor_fusion"];


          vehicle.prev_size = previous_path_x.size();
          if (vehicle.prev_size > 0)
          {
              vehicle.s = end_path_s;
          }

          vehicle.update(sensor_fusion);

          json msgJson;

          vector<double> next_x_vals;
          vector<double> next_y_vals;

          /**
           * TODO: define a path made up of (x,y) points that the car will visit
           *   sequentially every .02 seconds
           */

          /// evenly spaced at 30 meters
          vector<double> ptsx;
          vector<double> ptsy;

          /// reference x, y yaw states
          double ref_x = car_x;
          double ref_y = car_y;
          double ref_yaw = deg2rad(car_yaw);

          /// if previous point is almost empty, use the car as starting reference
          if (vehicle.prev_size < 2)
          {
            /// use two points that make the path tangent to the car
            double prev_car_x = car_x - cos(car_yaw);
            double prev_car_y = car_y - sin(car_yaw);

            ptsx.push_back(prev_car_x);
            ptsx.push_back(car_x);

            ptsy.push_back(prev_car_y);
            ptsy.push_back(car_y);
          }
          else
          {
            /// redefine reference state as previous path end point
            ref_x = previous_path_x[vehicle.prev_size - 1];
            ref_y = previous_path_y[vehicle.prev_size - 1];

            double ref_x_prev = previous_path_x[vehicle.prev_size - 2];
            double ref_y_prev = previous_path_y[vehicle.prev_size - 2];
            ref_yaw = atan2(ref_y - ref_y_prev, ref_x - ref_x_prev);

            ptsx.push_back(ref_x_prev);
            ptsx.push_back(ref_x);

            ptsy.push_back(ref_y_prev);
            ptsy.push_back(ref_y);
          }



          for (auto dist: {30, 60, 90})
          {
            double next_s = car_s + dist;
            double next_d = 2 + 4*vehicle.lane;
            auto xy = getXY(car_s + dist, next_d, map_waypoints_s, map_waypoints_x, map_waypoints_y);
            ptsx.push_back(xy[0]);
            ptsy.push_back(xy[1]);
          }

          for(int i = 0; i < ptsx.size(); ++i)
          {
            /// shift car reference angle to 0 degrees
            double shift_x = ptsx[i] - ref_x;
            double shift_y = ptsy[i] - ref_y;
            ptsx[i] = (shift_x * cos(0 - ref_yaw) - shift_y * sin(0 - ref_yaw));
            ptsy[i] = (shift_x * sin(0 - ref_yaw) + shift_y * cos(0 - ref_yaw));
          }

          tk::spline curve;
          curve.set_points(ptsx, ptsy);

          for (int i = 0; i < previous_path_x.size(); ++i)
          {
            next_x_vals.push_back(previous_path_x[i]);
            next_y_vals.push_back(previous_path_y[i]);
          }

          double target_x = 30.0;
          double target_y = curve(target_x);
          double target_dist = sqrt((target_x)*(target_x) + (target_y)*(target_y));

          double x_add_on = 0;
          for(int i = 0; i <= 50 - previous_path_x.size(); ++i)
          {
            double N = (target_dist / (0.02 * vehicle.ref_vel / 2.24));
            double x_point = x_add_on + target_x / N;
            double y_point = curve(x_point);
            x_add_on = x_point;

            double x_ref = x_point;
            double y_ref = y_point;

            // rotation
            x_point = (x_ref * cos(ref_yaw) - y_ref * sin(ref_yaw));
            y_point = (x_ref * sin(ref_yaw) + y_ref * cos(ref_yaw));

            // shift
            x_point += ref_x;
            y_point += ref_y;

            next_x_vals.push_back(x_point);
            next_y_vals.push_back(y_point);
          }

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          auto msg = "42[\"control\","+ msgJson.dump()+"]";

          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  
  h.run();
}