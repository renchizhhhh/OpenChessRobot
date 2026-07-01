"""Vision core for grasp-position correction (pure CV + board-plane geometry).

Shared by the standalone ``check_piece_offset.py`` diagnostic and the robot node
(``ChessRoboPlayer.measure_pick_offset``). No ROS dependency, so the detectors
are unit tested off-robot; thresholds are plain function arguments and callers
(the node / script) supply ROS-tunable values.

From a near-nadir ``look`` pose the lens sits straight above a square centre. The
pipeline is:

  1. ``detect_center_square`` - find the grid square at the image centre.
  2. ``detect_base_circle`` - fit the piece's round base inside that square.
  3. ``differential_offset_base`` - back-project the square centre and the base
     centre onto the board plane through the SAME live camera pose and subtract.
     Sharing one pose makes the hand-eye error common-mode, so it cancels and the
     result is a base-frame (x, y) offset that is hand-eye-independent to first
     order.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Detect the centre square of the chess grid                                   #
# --------------------------------------------------------------------------- #
def _line_at(seg, *, x=None, y=None):
    """Evaluate the infinite line through ``seg`` at a given x or y."""
    x1, y1, x2, y2 = seg
    if x is not None:
        if x2 == x1:
            return None
        return y1 + (x - x1) * (y2 - y1) / (x2 - x1)
    if y2 == y1:
        return None
    return x1 + (y - y1) * (x2 - x1) / (y2 - y1)


def _intersect(seg_a, seg_b):
    x1, y1, x2, y2 = seg_a
    x3, y3, x4, y4 = seg_b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return np.array([px, py])


def _angle_distance(a, b):
    """Smallest unsigned angle between two unoriented lines (radians)."""
    return abs((a - b + np.pi / 2) % np.pi - np.pi / 2)


def _dominant_segments(segments, max_angle_error=np.deg2rad(12.0)):
    """Keep segments belonging to the dominant direction in one line family.

    Hough often returns chair, table and piece edges in addition to grid lines.
    The grid produces several similarly-oriented segments, so a length-weighted
    doubled-angle mean gives a stable family direction and rejects isolated
    slanted edges.
    """
    if not segments:
        return [], None
    angles, weights = [], []
    for seg in segments:
        dx, dy = float(seg[2] - seg[0]), float(seg[3] - seg[1])
        angles.append(np.arctan2(dy, dx) % np.pi)
        weights.append(max(np.hypot(dx, dy), 1.0))
    angles = np.asarray(angles)
    weights = np.asarray(weights)
    mean_angle = 0.5 * np.arctan2(
        np.sum(weights * np.sin(2.0 * angles)),
        np.sum(weights * np.cos(2.0 * angles))) % np.pi
    keep = [seg for seg, angle in zip(segments, angles)
            if _angle_distance(angle, mean_angle) <= max_angle_error]
    if len(keep) < 2:
        return [], None

    # Re-estimate after removing the first-pass outliers.
    angles = np.array([
        np.arctan2(seg[3] - seg[1], seg[2] - seg[0]) % np.pi
        for seg in keep])
    weights = np.array([
        max(np.hypot(seg[2] - seg[0], seg[3] - seg[1]), 1.0)
        for seg in keep])
    mean_angle = 0.5 * np.arctan2(
        np.sum(weights * np.sin(2.0 * angles)),
        np.sum(weights * np.cos(2.0 * angles))) % np.pi
    direction = np.array([np.cos(mean_angle), np.sin(mean_angle)])
    return keep, direction


def _cluster_inner_edges(samples, tol, reference):
    """Merge a thick/frame border and retain its edge facing ``reference``.

    At an edge square the wooden frame and the playable-area boundary make two
    strong parallel lines roughly 20--25 px apart. Averaging those lines biases
    the measured square. For a group above/left of the image centre the largest
    coordinate is the inner edge; below/right it is the smallest coordinate.
    """
    groups = []
    for position, weight in sorted(samples):
        if not groups or position - groups[-1][-1][0] > tol:
            groups.append([[position, weight]])
        else:
            groups[-1].append([position, weight])

    positions, weights = [], []
    for group in groups:
        coordinates = np.asarray([item[0] for item in group], dtype=float)
        group_weights = np.asarray([item[1] for item in group], dtype=float)
        if np.all(coordinates < reference):
            position = float(np.max(coordinates))
        elif np.all(coordinates > reference):
            position = float(np.min(coordinates))
        else:
            position = float(np.average(coordinates, weights=group_weights))
        positions.append(position)
        weights.append(float(np.sum(group_weights)))
    return np.asarray(positions), np.asarray(weights)


def _fit_lattice(positions, weights, min_pitch, max_pitch,
                 residual_ratio=0.08):
    """Robustly fit ``position = phase + index * pitch`` to grid lines.

    Pairwise spacings are divided by plausible integer index differences.  A
    candidate wins by explaining many lines while leaving few holes.  That
    coverage term is important: half-pitch also fits every real line, but would
    predict a nonexistent line between every pair.
    """
    if len(positions) < 3:
        return None
    candidates = []
    for i in range(len(positions) - 1):
        for j in range(i + 1, len(positions)):
            distance = positions[j] - positions[i]
            for steps in range(1, 9):
                pitch = distance / steps
                if min_pitch <= pitch <= max_pitch:
                    candidates.append(pitch)
    if not candidates:
        return None

    best = None
    for pitch in candidates:
        tolerance = max(3.0, residual_ratio * pitch)
        for anchor in positions:
            phase = anchor % pitch
            residuals = np.abs((positions - phase + pitch / 2) % pitch - pitch / 2)
            inliers = residuals <= tolerance
            if np.count_nonzero(inliers) < 3:
                continue
            indices = np.rint((positions[inliers] - phase) / pitch).astype(int)
            index_span = int(np.ptp(indices))
            if index_span < 2:
                continue
            coverage = np.count_nonzero(inliers) / float(index_span + 1)
            weighted_support = float(np.sum(np.log1p(weights[inliers])))
            rms = float(np.sqrt(np.mean(residuals[inliers] ** 2)))
            # Lexicographic ranking strongly prefers explaining more actual
            # lines; coverage then rejects sub-harmonic (half-pitch) fits.
            rank = (np.count_nonzero(inliers), coverage, weighted_support,
                    -rms / pitch, pitch)
            if best is None or rank > best[0]:
                best = (rank, pitch, phase, inliers)
    if best is None:
        return None

    _, pitch, phase, inliers = best
    indices = np.rint((positions[inliers] - phase) / pitch).astype(int)
    design = np.column_stack((indices, np.ones_like(indices)))
    root_weights = np.sqrt(weights[inliers])
    fit, *_ = np.linalg.lstsq(
        design * root_weights[:, None],
        positions[inliers] * root_weights,
        rcond=None)
    pitch, phase = float(fit[0]), float(fit[1])
    if pitch <= 0:
        return None
    residuals = np.abs(positions - (phase + np.rint(
        (positions - phase) / pitch) * pitch))
    inliers = residuals <= max(3.0, residual_ratio * pitch)
    if np.count_nonzero(inliers) < 3:
        return None
    indices = np.rint((positions[inliers] - phase) / pitch).astype(int)
    if np.ptp(indices) < 2:
        return None
    return pitch, phase, inliers


def _line_through_axis(position, direction, centre, horizontal):
    """Make a long segment for an infinite fitted line at an axis crossing."""
    point = np.array([centre[0], position], dtype=float) if horizontal \
        else np.array([position, centre[1]], dtype=float)
    return np.r_[point - 10000.0 * direction,
                 point + 10000.0 * direction]


def _projection_peaks(edges, horizontal):
    """Find long axis-aligned edges deterministically by projection.

    ``HoughLinesP`` may randomly omit a weak pale grid boundary. Summing Canny
    support across a broad centre band consistently retains that boundary while
    rejecting short circular piece edges.
    """
    h, w = edges.shape
    cx, cy = w // 2, h // 2
    half_band = int(0.35 * min(h, w))
    if horizontal:
        band = edges[:, max(0, cx - half_band):min(w, cx + half_band)]
        profile = np.count_nonzero(band, axis=1)
    else:
        band = edges[max(0, cy - half_band):min(h, cy + half_band), :]
        profile = np.count_nonzero(band, axis=0)

    min_support = max(40, int(0.10 * min(h, w)))
    peaks = []
    for index in range(2, len(profile) - 2):
        value = profile[index]
        if value >= min_support and value == np.max(profile[index - 2:index + 3]):
            if not peaks or index - peaks[-1] > 3:
                peaks.append(index)
            elif value > profile[peaks[-1]]:
                peaks[-1] = index
    return peaks


def _bounds_around(reference, fit, positions, min_pitch, max_pitch):
    """Return lattice bounds, or the nearest credible observed pair."""
    if fit is not None:
        pitch, phase, _ = fit
        below = positions[positions < reference]
        above = positions[positions > reference]
        if len(below) and len(above):
            observed_lower = float(np.max(below))
            observed_upper = float(np.min(above))
            observed_pitch = observed_upper - observed_lower
            # Prefer the actual local edges when the lattice confirms they are
            # one pitch apart. This preserves mild perspective (the edge square
            # is a few pixels narrower than squares farther across the image).
            if abs(observed_pitch - pitch) / pitch <= 0.18:
                return (observed_lower, observed_upper, observed_pitch,
                        "observed-validated")
        lower = phase + np.floor((reference - phase) / pitch) * pitch
        return float(lower), float(lower + pitch), float(pitch), "lattice"

    below = positions[positions < reference]
    above = positions[positions > reference]
    if not len(below) or not len(above):
        return None
    lower, upper = float(np.max(below)), float(np.min(above))
    pitch = upper - lower
    if not min_pitch <= pitch <= max_pitch:
        return None
    return lower, upper, pitch, "observed"


def detect_center_square(gray, pitch_tol=0.08, aspect_tol=0.25,
                         center_tol=0.45, canny_low=35, canny_high=100):
    """Find the grid square containing the image centre.

    The look pose centres the lens over the square centre and views near-nadir.
    Long Hough segments are split into the two board directions, then each set
    is fitted to a regularly-spaced 1-D lattice.  The lattice fit, rather than
    the nearest four raw segments, is what makes this robust to duplicate Hough
    hits, unrelated straight edges, and one missing/occluded square boundary.
    ``pitch_tol`` is the maximum lattice residual as a fraction of one square.
    Fixed Canny thresholds preserve pale wood-on-wood grid boundaries; deriving
    them from median image brightness fails on washed-out white-piece captures.

    Returns ``(corners(4,2) [tl,tr,br,bl], centre(2), side_px)`` or ``None``.
    """
    if gray.ndim != 2:
        raise ValueError("detect_center_square expects a grayscale image")
    h, w = gray.shape
    cx, cy = w / 2.0, h / 2.0
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, int(canny_low), int(canny_high))
    cv2.setRNGSeed(0)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 360, threshold=max(50, min(h, w) // 14),
        minLineLength=max(40, min(h, w) // 5),
        maxLineGap=max(12, min(h, w) // 30))
    horiz, vert = [], []
    if lines is not None:
        for seg in lines.reshape(-1, 4):
            x1, y1, x2, y2 = seg
            ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
            if ang < 45 or ang > 135:
                horiz.append(seg)    # spans left-right -> bounds top/bottom
            else:
                vert.append(seg)     # spans top-bottom -> bounds left/right

    # Add deterministic long-edge evidence. These segments are deliberately
    # full-length: their coordinates carry the projection result while their
    # axis alignment stabilizes the dominant family direction.
    horiz.extend(np.array([0, y, w - 1, y])
                 for y in _projection_peaks(edges, horizontal=True))
    vert.extend(np.array([x, 0, x, h - 1])
                for x in _projection_peaks(edges, horizontal=False))

    horiz, h_direction = _dominant_segments(horiz)
    vert, v_direction = _dominant_segments(vert)
    if h_direction is None or v_direction is None:
        logger.warning("square rejected: no dominant horizontal/vertical grid lines")
        return None
    if abs(float(np.dot(h_direction, v_direction))) > np.sin(np.deg2rad(18.0)):
        logger.warning("square rejected: fitted line families are not perpendicular")
        return None

    def samples_at_axis(segments, horizontal):
        samples = []
        axis_limit = h if horizontal else w
        for seg in segments:
            pos = _line_at(seg, x=cx) if horizontal else _line_at(seg, y=cy)
            if pos is None or not (-0.2 * axis_limit <= pos <= 1.2 * axis_limit):
                continue
            length = float(np.hypot(seg[2] - seg[0], seg[3] - seg[1]))
            samples.append((float(pos), length))
        # A board-frame edge is much wider than a Canny duplicate. Collapse the
        # whole separator and retain the edge facing the target square.
        reference = cy if horizontal else cx
        return _cluster_inner_edges(
            samples, max(4.0, 0.03 * min(h, w)), reference)

    h_positions, h_weights = samples_at_axis(horiz, True)
    v_positions, v_weights = samples_at_axis(vert, False)
    min_pitch, max_pitch = 0.12 * min(h, w), 0.45 * min(h, w)
    h_fit = _fit_lattice(h_positions, h_weights, min_pitch, max_pitch, pitch_tol)
    v_fit = _fit_lattice(v_positions, v_weights, min_pitch, max_pitch, pitch_tol)
    h_bounds = _bounds_around(cy, h_fit, h_positions, min_pitch, max_pitch)
    v_bounds = _bounds_around(cx, v_fit, v_positions, min_pitch, max_pitch)
    if h_bounds is None or v_bounds is None:
        logger.warning(
            "square rejected: no credible borders around the image centre "
            "(found %d horizontal, %d vertical candidates)",
            len(h_positions), len(v_positions))
        return None
    top_y, bot_y, height, h_source = h_bounds
    left_x, right_x, width, v_source = v_bounds
    aspect_error = abs(width - height) / np.mean([width, height])
    if aspect_error > aspect_tol:
        logger.warning("square rejected: fitted pitch %.1f x %.1f px (aspect %.2f)",
                       width, height, width / height)
        return None
    if h_source == "observed" or v_source == "observed":
        logger.info(
            "centre square uses direct border fallback (%s rows, %s columns; "
            "%.1f x %.1f px)", h_source, v_source, width, height)

    image_centre = np.array([cx, cy])
    top = _line_through_axis(top_y, h_direction, image_centre, True)
    bottom = _line_through_axis(bot_y, h_direction, image_centre, True)
    left = _line_through_axis(left_x, v_direction, image_centre, False)
    right = _line_through_axis(right_x, v_direction, image_centre, False)

    tl = _intersect(top, left)
    tr = _intersect(top, right)
    br = _intersect(bottom, right)
    bl = _intersect(bottom, left)
    if any(c is None for c in (tl, tr, br, bl)):
        return None
    corners = np.array([tl, tr, br, bl])
    centre = corners.mean(axis=0)
    if (abs(centre[0] - cx) > center_tol * width or
            abs(centre[1] - cy) > center_tol * height):
        logger.warning("square rejected: camera is too close to a grid boundary")
        return None
    side_px = float(np.mean([
        np.linalg.norm(tr - tl), np.linalg.norm(br - tr),
        np.linalg.norm(bl - br), np.linalg.norm(tl - bl)]))
    return corners, centre, side_px


# --------------------------------------------------------------------------- #
# Detect the piece's round base inside the masked square                       #
# --------------------------------------------------------------------------- #
def _silhouette_base_circle(gray, corners, square_centre, side_px,
                            r_lo, r_hi, min_contrast=30.0):
    """Find the largest circular core inside a high-contrast piece silhouette.

    This is particularly useful for knights: the head extends the outer contour
    but cannot move the maximum of the silhouette's distance transform away
    from the broad round base. Returns ``(centre, radius)`` or ``(None, None)``.
    """
    x0, y0 = np.floor(np.min(corners, axis=0)).astype(int)
    x1, y1 = np.ceil(np.max(corners, axis=0)).astype(int)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(gray.shape[1] - 1, x1), min(gray.shape[0] - 1, y1)
    if x1 <= x0 or y1 <= y0:
        return None, None

    roi = gray[y0:y1 + 1, x0:x1 + 1]
    local_corners = corners - np.array([x0, y0])
    square_mask = np.zeros(roi.shape, np.uint8)
    cv2.fillConvexPoly(square_mask, local_corners.astype(np.int32), 255)
    mask_distance = cv2.distanceTransform(square_mask, cv2.DIST_L2, 5)

    # Estimate the bare square from a band near its perimeter, away from both
    # the grid line and the normally central piece.
    background_band = ((mask_distance > 0.04 * side_px) &
                       (mask_distance < 0.18 * side_px))
    if np.count_nonzero(background_band) < 50:
        return None, None
    background = float(np.median(roi[background_band]))
    difference = cv2.absdiff(
        roi, np.full(roi.shape, int(round(background)), dtype=np.uint8))
    difference = cv2.GaussianBlur(difference, (9, 9), 0)
    valid_values = difference[square_mask > 0]
    threshold, _ = cv2.threshold(
        valid_values.reshape(-1, 1), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    foreground = np.uint8(
        (difference > threshold) & (square_mask > 0)) * 255
    close_size = max(5, int(round(0.03 * side_px)) | 1)
    foreground = cv2.morphologyEx(
        foreground, cv2.MORPH_CLOSE,
        np.ones((close_size, close_size), np.uint8))
    foreground = cv2.morphologyEx(
        foreground, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground)
    local_square_centre = square_centre - np.array([x0, y0])
    components = [
        index for index in range(1, count)
        if 0.03 * side_px ** 2 <= stats[index, cv2.CC_STAT_AREA]
        <= 0.70 * side_px ** 2
    ]
    if not components:
        return None, None
    component_index = min(
        components,
        key=lambda index: np.linalg.norm(
            centroids[index] - local_square_centre))
    component = labels == component_index
    contrast = abs(float(np.mean(roi[component])) - background)
    if contrast < min_contrast:
        return None, None

    distance = cv2.distanceTransform(
        np.uint8(component) * 255, cv2.DIST_L2, 5)
    _, radius, _, centre_local = cv2.minMaxLoc(distance)
    centre = np.array(
        [centre_local[0] + x0, centre_local[1] + y0], dtype=float)
    if not (r_lo - 2.0 <= radius <= r_hi + 2.0):
        return None, None
    if np.linalg.norm(centre - square_centre) > 0.38 * side_px:
        return None, None
    return centre, float(radius)


def detect_base_circle(gray, corners, square_centre, side_px,
                       r_min_ratio=0.22, r_max_ratio=0.35,
                       clahe_clip=3.0, hough_param1=100,
                       hough_param2=27):
    """Fit the piece's round base inside the masked square.

    Mask to the detected square so neighbours / board are excluded, then Hough
    for the base ring. Hough (not the silhouette centroid or min-enclosing
    circle) is deliberate: an asymmetric top (knight) would bias a centroid, but
    the base is a clean circle that Hough locks onto, treating the head as
    outliers.

    A real base radius is ~22-35% of the square side, so candidates are constrained to
    ``[r_min_ratio, r_max_ratio] * side_px`` and any out-of-band fit (square
    border, finial ring, shadow) is rejected. Among in-band hits the one nearest
    the square centre wins. Returns ``(centre(2), radius)`` or ``(None, None)``.
    """
    r_lo, r_hi = r_min_ratio * side_px, r_max_ratio * side_px
    silhouette_centre, silhouette_radius = _silhouette_base_circle(
        gray, corners, square_centre, side_px, r_lo, r_hi)
    if silhouette_centre is not None:
        return silhouette_centre, silhouette_radius

    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)
    # Shrink the mask slightly so the bright square borders do not seed circles.
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8))
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    if clahe_clip > 0:
        masked = cv2.createCLAHE(
            clipLimit=float(clahe_clip), tileGridSize=(8, 8)).apply(masked)
        masked = cv2.bitwise_and(masked, masked, mask=mask)
    blur = cv2.medianBlur(masked, 5)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=0.12 * side_px,
        param1=float(hough_param1), param2=float(hough_param2),
        minRadius=int(np.floor(r_lo)), maxRadius=int(np.ceil(r_hi)))
    if circles is not None:
        # Hough works with integer radius limits but reports sub-pixel radii.
        # Keep a small numerical allowance at the exact band edge (the H8 pawn
        # is 71.4 px against a calculated lower bound of 71.5 px).
        in_band = [c for c in np.squeeze(circles, axis=0)
                   if r_lo - 2.0 <= c[2] <= r_hi + 2.0]
        if in_band:
            expected_radius = 0.27 * side_px
            best = min(
                in_band,
                key=lambda c: np.linalg.norm(c[:2] - square_centre) +
                0.35 * abs(c[2] - expected_radius))
            return np.array([best[0], best[1]]), float(best[2])
    # Fallback: largest contour inside the mask, accepted only if its radius is
    # in the expected base band.
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_and(thresh, mask)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        (bx, by), r = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
        if r_lo <= r <= r_hi:
            return np.array([bx, by]), float(r)
    return None, None


# --------------------------------------------------------------------------- #
# Board-plane geometry                                                         #
# --------------------------------------------------------------------------- #
def backproject(pixel, R_cam, cam_centre, K, dist, z_plane):
    """Intersect the camera ray through ``pixel`` with the plane ``z=z_plane``.

    ``R_cam`` / ``cam_centre`` are the camera-to-base rotation and lens position
    (see ``recognition_projection.camera_to_base``). Returns the (3,) base-frame
    point where the ray meets the plane.
    """
    pix = np.array([[[pixel[0], pixel[1]]]], dtype=np.float64)
    norm = cv2.undistortPoints(pix, K, dist).reshape(2)
    ray_base = R_cam.apply([norm[0], norm[1], 1.0])
    t = (z_plane - cam_centre[2]) / ray_base[2]
    return cam_centre + t * ray_base


def differential_offset_base(sq_centre, circ_centre, R_cam, cam_centre,
                             K, dist, board_z):
    """Base-frame piece-vs-square offset by differential back-projection.

    Both pixels are projected through the SAME camera pose onto ``z=board_z``,
    so the hand-eye error is common-mode and cancels in the difference. Returns
    a (2,) base-frame ``(dx, dy)`` offset in metres (piece centre minus square
    centre) to nudge the nominal grasp by.
    """
    p_sq = backproject(sq_centre, R_cam, cam_centre, K, dist, board_z)
    p_circ = backproject(circ_centre, R_cam, cam_centre, K, dist, board_z)
    return p_circ[:2] - p_sq[:2]


def draw_offset_annotation(img, corners, sq_centre, circ_centre=None,
                           radius=None):
    """Annotate a copy of ``img`` with the square, its centre and the base."""
    vis = img.copy()
    cv2.polylines(vis, [corners.astype(np.int32)], True, (0, 0, 255), 2)
    cv2.drawMarker(vis, tuple(np.asarray(sq_centre).astype(int)), (0, 0, 255),
                   cv2.MARKER_CROSS, 24, 2)
    if circ_centre is not None:
        centre = tuple(np.asarray(circ_centre).astype(int))
        if radius is not None:
            cv2.circle(vis, centre, int(radius), (0, 255, 0), 2)
        cv2.circle(vis, centre, 3, (0, 255, 0), -1)
        cv2.line(vis, tuple(np.asarray(sq_centre).astype(int)), centre,
                 (0, 255, 255), 2)
    return vis
