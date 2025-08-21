import cv2
import numpy as np


def detect_bezel_score_anycolor(frame):
    """
    베젤 색이 검정 or 흰 or 기타여도, '균일(단색)' 판단에 초점을 두고,
    가장자리 직선도(HoughLine)도 확인하여 score 산출
    return: score in [0..1] (higher => more suspicious of device bezel)
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # (A) Edge + HoughLine (유사)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=min(h,w)//4, maxLineGap=10)
    line_count = 0
    margin=10
    if lines is not None:
        for l in lines:
            x1,y1,x2,y2 = l[0]
            dx = abs(x2-x1)
            dy = abs(y2-y1)
            # 수직/수평
            if dx<5 or dy<5:
                # boundary 근접
                if (x1<margin or x2<margin or x1>w-margin or x2>w-margin or
                    y1<margin or y2<margin or y1>h-margin or y2>h-margin):
                    line_count+=1
    line_score = min(line_count/5, 1.0)
    
    # (B) boundary ROI uniform color check
    border_w = max(5, w//20)
    border_h = max(5, h//20)
    # top/bottom/left/right
    top    = frame[:border_h, :]
    bottom = frame[h-border_h:, :]
    left   = frame[:, :border_w]
    right  = frame[:, w-border_w:]
    
    # 전체 4개를 합쳐 1개 array로 처리
    # shape => (some_pixels, 3)
    boundary_pixels = np.concatenate([
        top.reshape(-1,3),
        bottom.reshape(-1,3),
        left.reshape(-1,3),
        right.reshape(-1,3),
    ], axis=0)
    
    # 색 편차 (stdB + stdG + stdR)
    std_bgr = np.std(boundary_pixels, axis=0)  # shape=(3,)
    sum_std = std_bgr.sum()                   # scalar
    
    # sum_std=0 => 완전 단색, sum_std> ~100 => 다양함
    # => uniform_score = 1 - (sum_std / 50)
    # (단순 예: sum_std=50이상이면 uniform_score=0, 0이면=1)
    uniform_score = max(0., 1. - sum_std/50.)
    
    # 최종 score = max(line_score, uniform_score)
    # => 어느 한쪽이 높아도 베젤 의심
    score = max(line_score, uniform_score)
    
    return max(0., min(score, 1.))  # clamp

    
def detect_bezel(frame, edge_threshold=100, black_threshold=40, coverage_ratio=0.5):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1) Canny + HoughLine
    edges = cv2.Canny(gray, 50, 150)
    # HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=edge_threshold,
                            minLineLength=min(h,w)//4, maxLineGap=10)

    line_count = 0
    if lines is not None:
        for l in lines:
            x1,y1,x2,y2 = l[0]
            # 수직/수평 판별 => slope
            dx = abs(x2-x1)
            dy = abs(y2-y1)
            # 예: 직선성 + near boundary
            if dx<5 or dy<5:
                # near boundary check?
                if x1<10 or x2<10 or x1>w-10 or x2>w-10 or y1<10 or y2<10 or y1>h-10 or y2>h-10:
                    line_count+=1
    
    # 2) Boundary region color check
    border_width = max(5, w//20)
    border_height= max(5, h//20)
    
    # top region
    top_region = gray[0:border_height, :]
    # bottom region
    bot_region = gray[h-border_height:h, :]
    # left region
    left_region= gray[:, 0:border_width]
    # right region
    right_region= gray[:, w-border_width:w]
    
    # 이 4개 영역에서 "매우 어두운(<= ~ 20) 픽셀"의 비율?
    dark_thresh = 30
    def dark_ratio(img_part):
        return np.mean(img_part <= dark_thresh)
    
    top_dark = dark_ratio(top_region)
    bot_dark = dark_ratio(bot_region)
    left_dark= dark_ratio(left_region)
    right_dark=dark_ratio(right_region)
    
    # average dark ratio
    mean_dark_ratio = (top_dark + bot_dark + left_dark + right_dark)/4
    
    # 결정
    # (예시) line_count>2 or mean_dark_ratio>0.5 => bezel suspected
    debug_info = (line_count, mean_dark_ratio)
    is_bezel = (line_count>2 or mean_dark_ratio>coverage_ratio)
    return is_bezel, debug_info


if __name__ == "__main__":
    path = '/purestorage/AILAB/AI_1/hkl/project/seeuon_web_demo/save_data/youngjun.hwang/camera_tablet/250226_002209_987560.jpg'
    frame = cv2.imread(path)
    # suspected, info = detect_bezel(frame)
    # print("Suspected bezel?", suspected, "debug=", info)

    sc = detect_bezel_score_anycolor(frame)
    print("bezel_score=", sc)