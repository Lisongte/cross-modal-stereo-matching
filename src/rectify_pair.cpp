#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

struct Args {
    std::string calib = "config/rgb_left_nir_right.yml";
    std::string left = "data/original_images/left_rgb.png";
    std::string right = "data/original_images/right_nir.png";
    std::string left_depth;
    std::string right_depth;
    std::string out_dir = "output/rectified";
    int width = 0;
    int height = 0;
    double alpha = 0.0;
};

static void print_usage(const char* argv0) {
    std::cerr
        << "Usage:\n"
        << "  " << argv0
        << " --calib config/rgb_left_nir_right.yml"
        << " --left data/original_images/left_rgb.png"
        << " --right data/original_images/right_nir.png"
        << " --out output/rectified [--left-depth DEPTH] [--right-depth DEPTH]"
        << " [--width W --height H --alpha A]\n";
}

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto need_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + name);
            }
            return argv[++i];
        };

        if (key == "--calib") {
            args.calib = need_value(key);
        } else if (key == "--left") {
            args.left = need_value(key);
        } else if (key == "--right") {
            args.right = need_value(key);
        } else if (key == "--left-depth") {
            args.left_depth = need_value(key);
        } else if (key == "--right-depth") {
            args.right_depth = need_value(key);
        } else if (key == "--out") {
            args.out_dir = need_value(key);
        } else if (key == "--width") {
            args.width = std::stoi(need_value(key));
        } else if (key == "--height") {
            args.height = std::stoi(need_value(key));
        } else if (key == "--alpha") {
            args.alpha = std::stod(need_value(key));
        } else if (key == "--help" || key == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    return args;
}

static cv::Mat read_required_matrix(cv::FileStorage& fs, const std::string& name) {
    cv::Mat m;
    fs[name] >> m;
    if (m.empty()) {
        throw std::runtime_error("missing calibration matrix: " + name);
    }
    m.convertTo(m, CV_64F);
    return m;
}

static cv::Mat make_overlay(const cv::Mat& left_rect, const cv::Mat& right_rect) {
    cv::Mat left_bgr;
    cv::Mat right_bgr;

    if (left_rect.channels() == 1) {
        cv::cvtColor(left_rect, left_bgr, cv::COLOR_GRAY2BGR);
    } else {
        left_bgr = left_rect.clone();
    }

    if (right_rect.channels() == 1) {
        cv::cvtColor(right_rect, right_bgr, cv::COLOR_GRAY2BGR);
    } else {
        right_bgr = right_rect.clone();
    }

    cv::Mat overlay;
    cv::addWeighted(left_bgr, 0.55, right_bgr, 0.45, 0.0, overlay);
    for (int y = 24; y < overlay.rows; y += 32) {
        cv::line(overlay, cv::Point(0, y), cv::Point(overlay.cols - 1, y),
                 cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
    }
    return overlay;
}

static void save_depth_vis(const cv::Mat& depth, const fs::path& path) {
    CV_Assert(depth.type() == CV_16U || depth.type() == CV_32F || depth.type() == CV_64F);

    cv::Mat valid = depth > 0;
    double min_v = 0.0;
    double max_v = 0.0;
    cv::minMaxLoc(depth, &min_v, &max_v, nullptr, nullptr, valid);
    if (max_v <= min_v) {
        return;
    }

    cv::Mat depth_32f;
    depth.convertTo(depth_32f, CV_32F);
    cv::Mat normalized;
    depth_32f.convertTo(normalized, CV_8U, 255.0 / (max_v - min_v), -255.0 * min_v / (max_v - min_v));
    normalized.setTo(0, ~valid);

    cv::Mat colored;
    cv::applyColorMap(normalized, colored, cv::COLORMAP_TURBO);
    colored.setTo(cv::Scalar(0, 0, 0), ~valid);
    cv::imwrite(path.string(), colored);
}

static void rectify_optional_depth(const std::string& input_path,
                                   const cv::Mat& mapx,
                                   const cv::Mat& mapy,
                                   const fs::path& output_path,
                                   const fs::path& vis_path) {
    if (input_path.empty()) {
        return;
    }

    cv::Mat depth = cv::imread(input_path, cv::IMREAD_UNCHANGED);
    if (depth.empty()) {
        throw std::runtime_error("failed to read depth image: " + input_path);
    }

    cv::Mat depth_rect;
    cv::remap(depth, depth_rect, mapx, mapy, cv::INTER_NEAREST, cv::BORDER_CONSTANT);
    cv::imwrite(output_path.string(), depth_rect);
    save_depth_vis(depth_rect, vis_path);
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);

        cv::Mat left = cv::imread(args.left, cv::IMREAD_COLOR);
        cv::Mat right = cv::imread(args.right, cv::IMREAD_GRAYSCALE);
        if (left.empty()) {
            throw std::runtime_error("failed to read left image: " + args.left);
        }
        if (right.empty()) {
            throw std::runtime_error("failed to read right image: " + args.right);
        }

        cv::FileStorage calib(args.calib, cv::FileStorage::READ);
        if (!calib.isOpened()) {
            throw std::runtime_error("failed to open calibration: " + args.calib);
        }

        cv::Mat K1 = read_required_matrix(calib, "K1");
        cv::Mat D1 = read_required_matrix(calib, "D1");
        cv::Mat K2 = read_required_matrix(calib, "K2");
        cv::Mat D2 = read_required_matrix(calib, "D2");
        cv::Mat R = read_required_matrix(calib, "R");
        cv::Mat T = read_required_matrix(calib, "T");

        cv::Size output_size(
            args.width > 0 ? args.width : left.cols,
            args.height > 0 ? args.height : left.rows);

        cv::Mat R1, R2, P1, P2, Q;
        cv::Rect roi1, roi2;
        cv::stereoRectify(
            K1, D1, K2, D2,
            left.size(),
            R, T,
            R1, R2, P1, P2, Q,
            cv::CALIB_ZERO_DISPARITY,
            args.alpha,
            output_size,
            &roi1, &roi2);

        cv::Mat map1x, map1y, map2x, map2y;
        cv::initUndistortRectifyMap(K1, D1, R1, P1, output_size, CV_32FC1, map1x, map1y);
        cv::initUndistortRectifyMap(K2, D2, R2, P2, output_size, CV_32FC1, map2x, map2y);

        cv::Mat left_rect, right_rect;
        cv::remap(left, left_rect, map1x, map1y, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        cv::remap(right, right_rect, map2x, map2y, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

        fs::create_directories(args.out_dir);
        const fs::path out_dir(args.out_dir);
        cv::imwrite((out_dir / "left_rgb_rectified.png").string(), left_rect);
        cv::imwrite((out_dir / "right_nir_rectified.png").string(), right_rect);
        cv::imwrite((out_dir / "overlay_epilines.png").string(), make_overlay(left_rect, right_rect));
        rectify_optional_depth(args.left_depth, map1x, map1y,
                               out_dir / "left_depth_rectified.png",
                               out_dir / "left_depth_rectified_vis.png");
        rectify_optional_depth(args.right_depth, map2x, map2y,
                               out_dir / "right_depth_rectified.png",
                               out_dir / "right_depth_rectified_vis.png");

        cv::FileStorage out((out_dir / "rectification.yml").string(), cv::FileStorage::WRITE);
        out << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
        out << "roi1_x" << roi1.x << "roi1_y" << roi1.y << "roi1_width" << roi1.width << "roi1_height" << roi1.height;
        out << "roi2_x" << roi2.x << "roi2_y" << roi2.y << "roi2_width" << roi2.width << "roi2_height" << roi2.height;
        out << "output_width" << output_size.width << "output_height" << output_size.height;

        std::cout << "left input:  " << left.cols << "x" << left.rows << "\n";
        std::cout << "right input: " << right.cols << "x" << right.rows << "\n";
        std::cout << "rectified:   " << output_size.width << "x" << output_size.height << "\n";
        std::cout << "wrote:       " << args.out_dir << "\n";
        std::cout << "P1 fx:       " << P1.at<double>(0, 0) << "\n";
        std::cout << "baseline:    " << std::abs(P2.at<double>(0, 3) / P2.at<double>(0, 0)) << " m\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }
    return 0;
}
