""" Just some utilities for running experiments. Unrelated to pytorch in general
"""

import math

def get_midpoint(tuple_1, tuple_2):
    # Takes in two tuples of floats (of arbitrary length) and returns their
    # midpoint
    return tuple([sum(_)/2.0 for _ in zip(tuple_1, tuple_2)])

def l2_dist(tuple_1, tuple_2):
    # Retuns the euclidean distance between two ordered pairs
    return math.sqrt(sum((x - y) **2 for x, y in zip(tuple_1, tuple_2)))

def level_sets_r2(oracle, original_point, discretization, tolerance=0.01,
                  convexity=None, x_axis_upper=None):
    """ Given a function, oracle: R^2 -> R, which takes in two-tuples, and an
        original two-tuple of the form (0, y), this will output a list of
        two-tuples of length (2 ** discretization + 1) where each point in the
        list should be an argument to the oracle which outputs a value within
        a tolerance proportion of the original_point. It's assumed oracle
        calls are expensive
    ARGS:
        oracle: function : (R, R) -> (R). Assumes monotonicity along any line
                           from the origin
        original point: tuple - (first index must be zero)
        discretization : int - number of recursive steps we do,
        tolerance : float - how close the level set points must be
        convexity : str or None - either 'concave', 'convex', or None. Just
                    a cute little tip to help us search more efficiently
        x_axis_upper : float - if not None is a maximum value for finding the
                               appropriate x-axis value
    RETURNS:
        list of two-tuples
    """

    ##########################################################################
    #   Assertions on inputs                                                 #
    ##########################################################################

    assert len(original_point) == 2

    assert original_point[0] == 0.0
    assert original_point[1] > 0.0
    assert isinstance(discretization, int)
    assert tolerance > 0
    if convexity is not None:
        assert convexity in ['convex', 'concave']

    tolerance_worthy = lambda r: 1 - tolerance < r < 1.0 + tolerance

    ##########################################################################
    #   Compute the original point and find x-axis point                     #
    ##########################################################################
    zero_value = oracle((0.0, 0.0))
    original_value = oracle(original_point)

    if original_value < zero_value: # FORCE MONOTONE UP FOR SIMPLICITY
        oracle = lambda tup: -1 * oracle(tup)
        zero_value = -1 * zero_value
        original_value = -1 * original_value


    # Find an upper bound for x
    x_guess_value = zero_value
    x_guess = 0.0
    if x_axis_upper is None:
        while x_guess_value < original_value:
            x_guess += original_point[1]
            x_guess_value = oracle((x_guess, 0))
        x_axis_upper = x_guess

    # Now find the right x via binary search
    x_guess = x_axis_upper / 2.0

    x_upper = x_axis_upper
    x_lower = 0
    x_guess = (x_upper + x_lower) / 2.0
    x_guess_value = oracle((x_guess, 0))
    ratio = x_guess_value / original_value

    while not tolerance_worthy(ratio):
        if x_guess_value < original_value: # if too low, move up
            x_lower = x_guess
            x_guess = (x_upper + x_guess) / 2.0
        else:
            x_upper = x_guess
            x_guess = (x_lower + x_guess) / 2.0
        x_guess_value = oracle((x_guess, 0))
        ratio = original_value / x_guess_value

    x_axis_point = (x_guess, 0)

    ##########################################################################
    #   Build recursive subroutine                                           #
    ##########################################################################

    def recursive_midpoint(left, right):
        """ Returns an output point with x-value as midpoint of left, right
            and binary searches in the vertical direction only
        """
        ######################################################################
        #   Assert things and set up definitions                             #
        ######################################################################

        x_l, y_l = left
        x_r, y_r = right
        assert x_r > x_l
        new_x = (x_l + x_r) / 2.0
        new_y = (y_l + y_r) / 2.0
        midpoint = (new_x, new_y)
        midpoint_val = oracle((new_x, new_y))

        if tolerance_worthy(midpoint_val / original_value):
            # in case we're already good
            return midpoint

        ######################################################################
        #   Determine bounding y's for new corner point                      #
        ######################################################################

        if midpoint_val < original_value: # walk away from origin
            # walk until we overshoot and then binsearch
            k = 0
            new_val = midpoint_val
            while new_val < original_value:

                new_guess = (midpoint[0],
                             midpoint[1] + 2 ** k)
                new_val = oracle(new_guess)
            upper = (new_guess[0], new_guess[1])
            lower = midpoint

        else: # walk towards origin
            lower = (0, 0)
            upper = midpoint

        ######################################################################
        #   Binsearch between upper and lower until we find good point       #
        ######################################################################

        new_val = midpoint_val
        new_guess = midpoint
        while not tolerance_worthy(new_val / original_value):
            if new_val < original_value:
                lower = new_guess
            else:
                upper = new_guess
            new_guess = get_midpoint(lower, upper)
            new_val = oracle(new_guess)

        return new_guess


    ##########################################################################
    #   Build list and return                                                #
    ##########################################################################
    current_list = [original_point, x_axis_point]
    for disc_step in xrange(discretization):
        print current_list
        print '\n'
        new_list = []
        for i in xrange(len(current_list) - 1):
            new_point = recursive_midpoint(current_list[i],
                                           current_list[i + 1])
            new_list.extend([current_list[i], new_point])

        new_list.append(current_list[-1])
        current_list = new_list
    return current_list



def equidistant_points(point_list, num_intervals):
    """ Takes in a list of ordered pairs and outputs a list of points along
        the piecewise linear path that the ordered pairs form. Then outputs
        (num_intervals + 1) numbers of points that are equally spaced along this
        line
    ARGS:
        point_list : tuple[] - list of tuples that are points in R^n
        num_points: int - number of 'intervals' we separate the path into
    RETURNS:
        list of length (num_intervals + 1) -- is the points that equidistantly
        separate the path
    """

    assert len(point_list) > 1
    total_length = sum(l2_dist(point_list[i], point_list[i + 1])
                       for i in xrange(len(point_list) - 1))

    interval_length = total_length / num_intervals

    remainder = 0
    output_list = [point_list[0]]
    waypoint_idx = 1
    for i in xrange(num_intervals + 1):
        dist_to_go = interval_length
        not_satisfied = True


        while not_satisfied:
            segment_length = l2_dist(point_list[waypoint_idx - 1],
                                     point_list[waypoint_idx])
            if segment_length - remainder <= dist_to_go:
                # case where we need to go to the next waypoint
                waypoint_idx += 1
                dist_to_go -= segment_length
                remainder = 0
            else:
                not_satisfied = False
                # interpolate point
                endpoint = point_list[waypoint_idx]
                startpoint = point_list[waypoint_idx - 1]

                direction = tuple((endpoint[i] - startpoint[i]) / segment_length
                                  for i in xrange(len(endpoint)))
                equipoint = tuple((dist_to_go + remainder) * direction[i] +
                                  startpoint[i] for i in xrange(len(startpoint))
                                 )
                output_list.append(equipoint)
                remainder += dist_to_go
            if waypoint_idx == len(point_list):
                output_list.append(point_list[-1])
                return output_list

    return output_list



'''
def bilinear_interpolated_isoline(heatmap, x_scale, y_scale, initial_y_point):
    """ Builds isolines by bilinear interpolation on a predetermined grid
    ARGS:
        heatmap: NxM array - 2D array of lattice points to perform bilinear
                             interpolation over. heatmap[i][j] means y=i, x=j
        x_scale : float - the maximum x-value
        y_scale : float - the maximum y-value
        initial_y_point: float - value between 0 and y_scale
    RETURNS:
        a list of (x,y) points scaled by the x_scale, y_scale
    """
    eps = 1e-6
    assert 0 <= initial_y_point <= y_scale
    y_len = len(heatmap)
    x_len = len(heatmap[0])

    x_interval = x_scale / float(x_len)
    y_interval = y_scale / float(y_len)

    def next_point(p, current_val):
        """ Gets the four corner points for a given point.
            TRIES TO GO RIGHT!

            This point better be on the top or left wall of a box. It will
            output a point that is on the top or left wall of a box
        """
        x_val, y_val = p
        x_idx = x_val / x_interval
        y_idx = y_val / y_interval

        y_idx_modulus = y_idx % 1.0
        x_idx_modulus = x_idx % 1.0

        # First rectify numericals to be in alignment with induction
        if (1 - eps < y_idx_modulus < 1):
            y_idx_modulus = 0


        # also handle corner cases


        # Either on the left side, diagonal, or top
        if x_idx_modulus < eps or :
            left_wall = True
            diag = False

        elif abs(x_idx_modulus - y_idx_modulus) < eps:
            diag = True
            left_wall = False
        else:
            raise Exception("BROKE INDUCTIVE ASSUMPTION")

        left_x, right_x = int(math.floor(x_idx)), int(math.floor(x_idx) + 1)
        lower_y, upper_y = int(math.floor(y_idx)), int(math.floor(y_idx) + 1)
        ll_val = heatmap[lower_y][left_x]
        ul_val = heatmap[upper_y][left_x]
        lr_val = heatmap[lower_y][right_x]
        ur_val = heatmap[upper_y][right_x]

        if left_wall
            # compute the plane considered by the lower left triangle
            plane_eqn = [lr_val - ll_val, ll_val-ul_val, -1]
            direction = [plane_eqn[1], -plane_eqn[0]] # <x, y>
            # find the iso direction that points right
            if direction[0] < 0:
                direction = [-1 * _ for _ in direction]
            elif direction[0] == 0:
                if direction[1] > 0:
                    direction[1] = -1 * direction[1]

            # determine if we hit x-axis or x=-y line:
            to_lr_corner_perp = [-y_idx_modulus, -1]
            if sum(to_lr_corner_perp[i] * direction[i] for i in xrange(2)) > 0:
                x_axis_intersection = True
            else:
                x_axis_intersection = False

            if x_axis_intersection:
                new_point = (y_idx_modulus * direction[0]/ -direction[1],
                             y_interval * lower_y)
        else:



            to_x_axis_steps = y_idx_modulus / direction[1]
            direction[1] * to_x_axis_steps








    current_point = (0, initial_y_point)

'''






