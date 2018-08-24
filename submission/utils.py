class Box:
    def __init__(self, class_id, score, x_min, y_min, x_max, y_max):
        self.class_id = class_id
        self.score = score
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max


def parse_prediction_line(line, w=1, h=1):
    predictions = []
    for i in range(0, len(line), 6):
        predictions += [Box(
            class_id=line[i],
            score=float(line[i+1]),
            x_min=int(float(line[i + 2]) * w),
            y_min=int(float(line[i + 3]) * h),
            x_max=int(float(line[i + 4]) * w),
            y_max=int(float(line[i + 5]) * h),
        )]
    return predictions
