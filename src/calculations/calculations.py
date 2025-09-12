import numpy as np

def get_label_centroid(img, label=1, cv_format=False, is_transposed=True):
    """Function to calculate the centroid of a label in an image
    Args:
        img (np.array): Image to calculate the centroid
        label (int, optional): Label to calculate the centroid. Defaults to 1.
        cv_format (bool, optional): If True, the y-axis is inverted. Defaults to False.
        is_transposed (bool, optional): If True, the image is transposed. Defaults to True.
    Returns:
        (tuple): Centroid coordinates (x, y) or None if the label is not found
    """
    y_coords, x_coords = np.where(img == label)

    if len(x_coords) == 0 or len(y_coords) == 0:
        return None  
    
    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)

    y = int(is_transposed)

    if cv_format != False:
        centroid_y = img.shape[y] - centroid_y
    
    return centroid_x, centroid_y


def get_ventricle_function(image, cv_format=False, is_transposed=True, perpendicular=False):
    """Function to calculate the line that crosses the ventricles
    Args:
        image (np.array): Image to calculate the line
        cv_format (bool, optional): If True, the y-axis is inverted. Defaults to False.
        is_transposed (bool, optional): If True, the image is transposed. Defaults to True.
        perpendicular (bool, optional): If True, returns the perpendicular line to the ventricles line. Defaults to False.
    Returns:
        (tuple): Slope (m) and y-intercept (b) of the line
    """
    centroid_rv = get_label_centroid(image, 1, cv_format=cv_format, is_transposed=is_transposed)
    centroid_lv = get_label_centroid(image, 3, cv_format=cv_format, is_transposed=is_transposed)

    if centroid_rv is None or centroid_lv is None:
        return None

    centroid_rv_x, centroid_rv_y = centroid_rv
    centroid_lv_x, centroid_lv_y = centroid_lv

    m = (centroid_rv_y - centroid_lv_y) / (centroid_rv_x - centroid_lv_x)
    b = centroid_rv_y - m * centroid_rv_x

    if perpendicular:
        m = -1/m
        b = centroid_rv_y - m * centroid_rv_x

        return m, b
    return m, b


def get_perpendicular_lv_function(image, cv_format=False, is_transposed=True):
    """Function to calculate the line perpendicular to the ventricles line that crosses the right ventricle
    Args:
        image (np.array): Image to calculate the line
        cv_format (bool, optional): If True, the y-axis is inverted. Defaults to False.
        is_transposed (bool, optional): If True, the image is transposed. Defaults to True.
    Returns:
        (tuple): Slope (m) and y-intercept (b) of the perpendicular line.    
    """
    centroid_rv = get_label_centroid(image, 1, cv_format=cv_format, is_transposed=is_transposed)
    m, _ = get_ventricle_function(image, cv_format=cv_format, is_transposed=is_transposed)
    m_perpendicular = -1/m

    if centroid_rv is None:
        return None
    
    centroid_rv_x, centroid_rv_y = centroid_rv
    b_perpendicular = centroid_rv_y - (m_perpendicular * centroid_rv_x)

    return m_perpendicular, b_perpendicular


def get_limits(height, width, m, b):
    """Function to calculate the intersection points of a line with the image borders
    Args:
        height (int): Height of the image
        width (int): Width of the image
        m (float): Slope of the line
        b (float): Y-intercept of the line
    Returns:
        (list): List of intersection points (x, y)
    """
    x_left, y_left = 0, b
    x_right, y_right = width - 1, m * (width - 1) + b
    x_top, y_top = -b / m, 0 
    x_bottom, y_bottom = (height - 1 - b) / m, height - 1
    points = []
    if 0 <= y_left < height:
        points.append((x_left, int(y_left)))
    if 0 <= y_right < height:
        points.append((x_right, int(y_right)))
    if 0 <= x_top < width:
        points.append((int(x_top), y_top))
    if 0 <= x_bottom < width:
        points.append((int(x_bottom), y_bottom))
    return points

def get_limits_2(height, width, m, b):
    """Function to calculate the intersection points of a line with the image borders
    Args:
        height (int): Height of the image
        width (int): Width of the image
        m (float): Slope of the line
        b (float): Y-intercept of the line
    Returns:
        (list): List of intersection points (x, y)
    """
    points = set()  # Use a set to avoid duplicates
    y_left = b
    if 0 <= y_left < height:
        points.add((0, int(y_left)))
    y_right = m * (width - 1) + b
    if 0 <= y_right < height:
        points.add((width - 1, int(y_right)))
    if m != 0:
        x_top = -b / m
        if 0 <= x_top < width:
            points.add((int(x_top), 0))
    if m != 0:
        x_bottom = (height - 1 - b) / m
        if 0 <= x_bottom < width:
            points.add((int(x_bottom), height - 1))
    points = list(points)
    if len(points) > 2:
        points = sorted(points)[:2] 

    return points