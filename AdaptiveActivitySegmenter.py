# AdaptiveActivitySegmenter
class AdaptiveActivitySegmenter:
    def __init__(self, window_size=10, base_threshold=1.5, feature_size=1):
        self.window_size = window_size
        self.base_threshold = base_threshold
        self.feature_size = feature_size
        self.data_window = np.zeros((window_size, feature_size))
        self.current_pos = 0
        self.last_mean = None
        self.variances = np.zeros(window_size)
        print(f"AdaptiveActivitySegmenter initialized with feature size: {feature_size}")

    def rolling_median(self, new_data_point):
        if new_data_point.shape != (self.feature_size,):
            raise ValueError(
                f"New data point shape {new_data_point.shape} does not match expected shape ({self.feature_size},)"
            )

        index = self.current_pos % self.window_size
        self.data_window[index] = new_data_point
        self.current_pos += 1

        window_data = self.data_window[:min(self.current_pos, self.window_size)]
        return np.median(window_data, axis=0)

    def detect_change_point(self, new_data_point):
        try:
            filtered_point = self.rolling_median(new_data_point)

            if self.current_pos < self.window_size:
                return False

            current_mean = np.mean(self.data_window, axis=0)
            current_variance = np.var(self.data_window, axis=0)
            self.variances[self.current_pos % self.window_size] = np.mean(current_variance)

            dynamic_threshold = self.base_threshold + np.mean(self.variances)
            if self.last_mean is not None and np.linalg.norm(current_mean - self.last_mean) > dynamic_threshold:
                self.last_mean = current_mean
                return "Change point detected"

            self.last_mean = current_mean
            return False
        except ValueError as e:
            print(f"Error detecting change point: {e}")
            return False
