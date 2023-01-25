'''
Adopted from
https://github.com/StanfordVL/iGibson/blob/master/igibson/external/pybullet_tools/utils.py
'''

from __future__ import print_function

import pybullet as p
from collections import defaultdict, deque, namedtuple
from itertools import product, combinations, count
import os, sys
import itertools


BASE_LINK = -1
MAX_DISTANCE = 0.

def path_starting_from_code(go_back_n_times: int = 0) -> str:
    """
    Return the path of the parent directory of where the code is.
    Can also going backward to return an upper directory

    Input:
        - int : how much you want to go backward

    Output:
        - str : path of the wanted directory
    
    """
    path = os.path.dirname(__file__) # parent directory of where the program resides
    for _ in itertools.repeat(None, go_back_n_times):
        path = os.path.dirname(path) # going back in time, parent direction of the parent..;
    return path



def pairwise_link_collision(body1, link1, body2, link2=BASE_LINK, max_distance=MAX_DISTANCE):  # 10000
    return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance,
                                  linkIndexA=link1, linkIndexB=link2)) != 0  # getContactPoints

def pairwise_collision(body1, body2, **kwargs):
    """
    The robot is describes as: [robot_id, [robot_links]]
    Check for each links the collision with the body 2
    """
    if isinstance(body1, tuple) or isinstance(body2, tuple):
        body1, links1 = expand_links(body1)
        body2, links2 = expand_links(body2)
        return any_link_pair_collision(body1, links1, body2, links2, **kwargs)
    return body_collision(body1, body2, **kwargs)

def expand_links(body):
    body, links = body if isinstance(body, tuple) else (body, None)
    if links is None:
        links = get_all_links(body)
    return body, links

def any_link_pair_collision(body1, links1, body2, links2=None, **kwargs):
    """
    
    """

    # TODO: this likely isn't needed anymore
    if links1 is None:
        links1 = get_all_links(body1)
    if links2 is None:
        links2 = get_all_links(body2)
    for link1, link2 in product(links1, links2):
        if (body1 == body2) and (link1 == link2):
            continue
        if pairwise_link_collision(body1, link1, body2, link2, **kwargs):
            # print('body {} link {} body {} link {}'.format(body1, link1, body2, link2))
            return True
    return False

def body_collision(body1, body2, max_distance=MAX_DISTANCE):  # 10000
    """
    collision if distance < MAX_DISTANCE
    """
    return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance)) != 0  # getContactPoints`

def get_self_link_pairs(body, joints, disabled_collisions=set(), only_moving=True):
    """
    
    """

    moving_links = get_moving_links(body, joints)
    fixed_links = list(set(get_joints(body)) - set(moving_links))
    check_link_pairs = list(product(moving_links, fixed_links))

    if only_moving:
        check_link_pairs.extend(get_moving_pairs(body, joints))
    else:
        check_link_pairs.extend(combinations(moving_links, 2))

    check_link_pairs = list(
        filter(lambda pair: not are_links_adjacent(body, *pair), check_link_pairs))
    check_link_pairs = list(filter(lambda pair: (pair not in disabled_collisions) and
                                                (pair[::-1] not in disabled_collisions), check_link_pairs))
    return check_link_pairs

def get_moving_links(body, joints):
    moving_links = set()
    for joint in joints:
        link = child_link_from_joint(joint)
        if link not in moving_links:
            moving_links.update(get_link_subtree(body, link))
    return list(moving_links)

def get_moving_pairs(body, moving_joints):
    """
    Check all fixed and moving pairs
    Do not check all fixed and fixed pairs
    Check all moving pairs with a common
    """
    moving_links = get_moving_links(body, moving_joints)
    for link1, link2 in combinations(moving_links, 2):
        ancestors1 = set(get_joint_ancestors(body, link1)) & set(moving_joints)
        ancestors2 = set(get_joint_ancestors(body, link2)) & set(moving_joints)
        if ancestors1 != ancestors2:
            yield link1, link2


#####################################

JointInfo = namedtuple('JointInfo', ['jointIndex', 'jointName', 'jointType',
                                     'qIndex', 'uIndex', 'flags',
                                     'jointDamping', 'jointFriction', 'jointLowerLimit', 'jointUpperLimit',
                                     'jointMaxForce', 'jointMaxVelocity', 'linkName', 'jointAxis',
                                     'parentFramePos', 'parentFrameOrn', 'parentIndex'])

def get_joint_info(body, joint):
    """
    Joint Info but named
    """
    return JointInfo(*p.getJointInfo(body, joint))

def child_link_from_joint(joint):
    return joint  # link

def get_num_joints(body):
    """
    Return number of joints (all)
    """
    return p.getNumJoints(body)

def get_joints(body):
    """
    List of all joints
    """
    return list(range(get_num_joints(body)))

get_links = get_joints

def get_all_links(body):
    """
    List of all links
    """
    return [BASE_LINK] + list(get_links(body))

def get_link_parent(body, link):
    """
    Get parent link of link from the body
    """
    if link == BASE_LINK:
        return None
    return get_joint_info(body, link).parentIndex

def get_all_link_parents(body):
    """
    Return a dictionnary with the parent link of all links as:
    {children_link: parent_link}
    """
    return {link: get_link_parent(body, link) for link in get_links(body)}

def get_all_link_children(body):
    """
    Return a dictionnary with the parent link of all links as:
    {parent_link: children_links}

    A parent link can have several children links
    """
    children = {}

    for child, parent in get_all_link_parents(body).items():

        if parent not in children:
            children[parent] = []

        children[parent].append(child)

    return children

def get_link_children(body, link):
    """
    Get the children link of link in the dictionnary

    Default value is []
    """
    children = get_all_link_children(body)

    return children.get(link, []) 



def get_link_ancestors(body, link):
    # Returns in order of depth
    # Does not include link
    parent = get_link_parent(body, link)

    if parent is None:
        return []

    return get_link_ancestors(body, parent) + [parent]


def get_joint_ancestors(body, joint):
    link = child_link_from_joint(joint)
    return get_link_ancestors(body, link) + [link]

def get_link_descendants(body, link, test=lambda l: True):
    """
    Get all descendant from a link not only the children
    """
    descendants = []

    for child in get_link_children(body, link):

        if test(child):
            descendants.append(child)
            descendants.extend(get_link_descendants(body, child, test=test))

    return descendants


def get_link_subtree(body, link, **kwargs):
    """
    Get link subtree 
    Return [link + all descendants]
    """

    return [link] + get_link_descendants(body, link, **kwargs)

def are_links_adjacent(body, link1, link2):

    return (get_link_parent(body, link1) == link2) or \
           (get_link_parent(body, link2) == link1)

