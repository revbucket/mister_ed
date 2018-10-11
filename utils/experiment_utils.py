""" Just some utilities for running experiments. Unrelated to pytorch in general
"""
from __future__ import print_function
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
    for disc_step in range(discretization):
        print(current_list)
        print('\n')
        new_list = []
        for i in range(len(current_list) - 1):
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
                       for i in range(len(point_list) - 1))

    interval_length = total_length / num_intervals

    remainder = 0
    output_list = [point_list[0]]
    waypoint_idx = 1
    for i in range(num_intervals + 1):
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
                                  for i in range(len(endpoint)))
                equipoint = tuple((dist_to_go + remainder) * direction[i] +
                                  startpoint[i] for i in range(len(startpoint))
                                 )
                output_list.append(equipoint)
                remainder += dist_to_go
            if waypoint_idx == len(point_list):
                output_list.append(point_list[-1])
                return output_list

    return output_list






