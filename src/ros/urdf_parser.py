import xml.etree.ElementTree as ET
import trep
from math import sin, cos, pi
import numpy as np
from numpy.linalg import norm

def transform(xyz, rpy):
    r = rpy[0];
    p = rpy[1];
    y = rpy[2];
   
    return np.array([[cos(y)*cos(p), cos(y)*sin(p)*sin(r)+sin(y)*cos(r), -cos(y)*sin(p)*cos(r)+sin(y)*sin(r)],
            [-sin(y)*cos(p), -sin(y)*sin(p)*sin(r)+cos(y)*cos(r), sin(y)*sin(p)*cos(r)+cos(y)*sin(r)],
            [sin(p), -cos(p)*sin(r), cos(p)*cos(r)],
            xyz])

def rotate(axis):
    A1 = np.array([1,0,0])
    A2 = np.array(axis)
    v = np.cross(A1, A2)
    c = np.dot(A1, A2)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    
    return np.eye(3) + vx + np.dot(vx,vx)*((1-c)/(norm(v)**2))
    
def add_child_frame(parent_name, parent_frame, links, joints, prefix):
    for joint in joints:
        if joint.find('parent').get('link') == parent_name:
            child_name = joint.find('child').get('link')
            joint_name = joint.get('name')

            # check if joint should be kinematically controlled
            if joint.get('kinematic') == 'True':
                iskin = True
            else:
                iskin = False
            
            # First create new joint frame if origin tag is specified
            j_origin = joint.find('origin')
            if j_origin == None:
                joint_frame = parent_frame
            else:
                if j_origin.get('xyz') is not None:
                    j_xyz = str.split(j_origin.get('xyz'))
                else:
                    j_xyz = [0.0, 0.0, 0.0]
                if j_origin.get('rpy') is not None:
                    j_rpy = str.split(j_origin.get('rpy'))
                else:
                    j_rpy = [0.0, 0.0, 0.0]
                origin = transform([float(j_xyz[0]),float(j_xyz[1]),float(j_xyz[2])],[float(j_rpy[0]),float(j_rpy[1]),float(j_rpy[2])])
                joint_frame = trep.Frame(parent_frame, trep.CONST_SE3, origin, name = prefix + joint_name)
            
            # Fixed Joint
            if joint.get('type') == 'fixed':
                child_frame = joint_frame
                child_frame.name = prefix + child_name
            
            # Continuous Joint
            elif joint.get('type') == 'continuous':
                j_axis = joint.find('axis')
                if j_axis == None:
                    j_xyz = [1, 0, 0]
                else:
                    j_xyz = str.split(j_axis.get('xyz'))
                    j_xyz = [float(j_xyz[0]), float(j_xyz[1]), float(j_xyz[2])]
                    j_xyz /= norm(j_xyz)
                    
                if np.dot(j_xyz, [1,0,0]) > 0.99:
                    #rotate about x-axis
                    child_frame = trep.Frame(joint_frame, trep.RX, prefix + joint_name, name = prefix + child_name, kinematic = iskin) 
                    
                elif np.dot(j_xyz, [0,1,0]) > 0.99:
                    #rotate about y-axis
                    child_frame = trep.Frame(joint_frame, trep.RY, prefix + joint_name, name = prefix + child_name, kinematic = iskin) 
                    
                elif np.dot(j_xyz, [0,0,1]) > 0.99:
                    #rotate about z-axis
                    child_frame = trep.Frame(joint_frame, trep.RZ, prefix + joint_name, name = prefix + child_name, kinematic = iskin) 
                    
                else:
                    #arbitray axis rotation
                    rotation_mat = rotate(j_xyz)
                    rotation_frame = trep.Frame(joint_frame, trep.CONST_SE3, np.vstack((rotation_mat,[0,0,0])), name = prefix + child_name + '-axis')
                    config_frame = trep.Frame(rotation_frame, trep.RX, prefix + joint_name, name = prefix + joint_name + '-axis', kinematic = iskin)
                    child_frame = trep.Frame(config_frame, trep.CONST_SE3, np.vstack((np.transpose(rotation_mat),[0,0,0])), name = prefix + child_name)
             
            # Prismatic Joint
            elif joint.get('type') == 'prismatic':                  
                j_axis = joint.find('axis')
                if j_axis == None:
                    j_xyz = [1, 0, 0]
                else:
                    j_xyz = str.split(j_axis.get('xyz'))
                    j_xyz = [float(j_xyz[0]), float(j_xyz[1]), float(j_xyz[2])]
                    j_xyz /= norm(j_xyz)
                    
                if np.dot(j_xyz, [1,0,0]) > 0.99:
                    #translate on x-axis
                    child_frame = trep.Frame(joint_frame, trep.TX, prefix + joint_name, name = prefix + child_name, kinematic = iskin) 
                    
                elif np.dot(j_xyz, [0,1,0]) > 0.99:
                    #translate on y-axis
                    child_frame = trep.Frame(joint_frame, trep.TY, prefix + joint_name, name = prefix + child_name, kinematic = iskin) 
                    
                elif np.dot(j_xyz, [0,0,1]) > 0.99:
                    #translate on z-axis
                    child_frame = trep.Frame(joint_frame, trep.TZ, prefix + joint_name, name = prefix + child_name, kinematic = iskin) 
                    
                else:
                    #arbitray axis translation
                    rotation_mat = rotate(j_xyz)
                    translation_frame = trep.Frame(joint_frame, trep.CONST_SE3, np.vstack((rotation_mat,[0,0,0])), name = prefix + child_name + '-axis')
                    config_frame = trep.Frame(translation_frame, trep.TX, prefix + joint_name, name = prefix + joint_name + '-axis', kinematic = iskin)
                    child_frame = trep.Frame(config_frame, trep.CONST_SE3, np.vstack((np.transpose(rotation_mat),[0,0,0])), name = prefix + child_name)
                    
            # Floating Joint - may not enable since robot_state_pubisher doesn't support
            # elif joint.get('type') == 'floating':
                # tx_frame = trep.Frame(joint_frame, trep.TX, joint_name + '-TX') 
                # ty_frame = trep.Frame(tx_frame, trep.TY, joint_name + '-TY') 
                # tz_frame = trep.Frame(ty_frame, trep.TZ, joint_name + '-TZ') 
                # rx_frame = trep.Frame(tz_frame, trep.RX, joint_name + '-RX') 
                # ry_frame = trep.Frame(rx_frame, trep.RY, joint_name + '-RY')
                # child_frame = trep.Frame(ry_frame, trep.RZ, child_name)     
            
            # No match for joint type
            else:
                print "Invalid joint type specified. Exiting..."
                break
            
            # add mass to child link
            inertia_offset = links[child_name]['origin']['xyz']
            inertia_rotate = links[child_name]['origin']['rpy']
            if norm(inertia_offset) + norm(inertia_rotate) > 0.001:
                # need to create link inertial frame
                inertial_frame = trep.Frame(child_frame, trep.CONST_SE3, transform(inertia_offset, inertia_rotate), name = prefix + child_name + '-inertia')
                inertial_frame.set_mass(links[child_name]['mass']['value'], Ixx = links[child_name]['inertia']['ixx'], Iyy = links[child_name]['inertia']['iyy'], Izz = links[child_name]['inertia']['izz'])
            else:
                child_frame.set_mass(links[child_name]['mass']['value'], Ixx = links[child_name]['inertia']['ixx'], Iyy = links[child_name]['inertia']['iyy'], Izz = links[child_name]['inertia']['izz'])
            
            # Start child node recursion
            add_child_frame(child_name, child_frame, links, joints, prefix)

def import_urdf_file(filename, system=None, prefix=None):
    tree = ET.parse(filename)
    root = tree.getroot()
    return load_urdf(root, system, prefix)
   
def import_urdf(source, system=None, prefix=None):
    root = ET.XML(source)
    return load_urdf(root, system, prefix)

def load_urdf(root, system=None, prefix=None):
    if system is None:
        system = trep.System()
    
    if prefix is None:
        prefix = ''

    links = root.findall('link')
    joints = root.findall('joint')

    link_dict = {}
    for link in links:
        inertial = link.find('inertial')
        if inertial is not None:
            inertia = inertial.find('inertia')
            if inertia is not None:
                inertia_dict = inertia.attrib
                for attr in ['ixx','iyy','izz']:
                    if attr in inertia_dict:
                        inertia_dict[attr] = float(inertia_dict[attr])
                    else:
                        inertia_dict[attr] = 0.0
            else:
                inertia_dict = {'ixx':0.0, 'iyy':0.0, 'izz':0.0} 
            origin = inertial.find('origin')
            if origin is not None:
                origin_dict = origin.attrib
                for attr in ['xyz', 'rpy']:
                    if attr in origin_dict:
                        values = str.split(origin_dict[attr])
                        origin_dict[attr] = [float(values[0]),float(values[1]),float(values[2])]
                    else:
                        origin_dict[attr] = [0.0, 0.0, 0.0]                    
            else:
                origin_dict = {'xyz':[0.0, 0.0, 0.0], 'rpy':[0.0, 0.0, 0.0]}   
            mass = inertial.find('mass')
            if mass is not None:
                mass_dict = mass.attrib
                if 'value' in mass_dict:
                    mass_dict['value'] = float(mass_dict['value'])
                else:
                    mass_dict['value'] = 0.0
            else:
                mass_dict = {'value':0.0}
            inertial_dict = {'origin':origin_dict, 'mass':mass_dict, 'inertia':inertia_dict}
        else:
            inertial_dict = {'origin':{'xyz':[0.0, 0.0, 0.0], 'rpy':[0.0, 0.0, 0.0]}, 'mass':{'value':0.0}, 'inertia':{'ixx':0.0, 'iyy':0.0, 'izz':0.0}}
            
        link_dict[link.get('name')] = inertial_dict

    root_link = []
    for link in links:
        name = link.get('name')

        ischild = False
        for joint in joints:
            if joint.find('child').get('link') == name:
                ischild = True
                break
        if not ischild:
            root_link.append(name)

    frames = []
    root_name = root_link[0]

    # check if root is world and add all frames
    if root_name == 'world':
        add_child_frame(root_name, system.world_frame, link_dict, joints, prefix)

    # else, must first create floating link to world_frame then add frames
    else:
        robot_name = root.get('name')
        world_tx = trep.Frame(system.world_frame, trep.TX, prefix + robot_name +'-TX', name = prefix + robot_name +'-TX')
        world_ty = trep.Frame(world_tx, trep.TY, prefix + robot_name +'-TY', name = prefix + robot_name +'-TY') 
        world_tz = trep.Frame(world_ty, trep.TZ, prefix + robot_name +'-TZ', name = prefix + robot_name +'-TZ') 
        world_rx = trep.Frame(world_tz, trep.RX, prefix + robot_name +'-RX', name = prefix + robot_name +'-RX')  
        world_ry = trep.Frame(world_rx, trep.RY, prefix + robot_name +'-RY', name = prefix + robot_name +'-RY')  
        root_frame = trep.Frame(world_ry, trep.RZ, prefix + robot_name +'-RZ', name = prefix + root_name)  

        # add mass to root frame
        inertia_offset = link_dict[root_name]['origin']['xyz']
        inertia_rotate = link_dict[root_name]['origin']['rpy']
        if norm(inertia_offset) + norm(inertia_rotate) > 0.001:
            # need to create link inertial frame
            inertial_frame = trep.Frame(root_frame, trep.CONST_SE3, transform(inertia_offset, inertia_rotate), name = prefix + root_name + '-inertia')
            inertial_frame.set_mass(link_dict[root_name]['mass']['value'], Ixx = link_dict[root_name]['inertia']['ixx'], Iyy = link_dict[root_name]['inertia']['iyy'], Izz = link_dict[root_name]['inertia']['izz'])
        else:
            root_frame.set_mass(link_dict[root_name]['mass']['value'], Ixx = link_dict[root_name]['inertia']['ixx'], Iyy = link_dict[root_name]['inertia']['iyy'], Izz = link_dict[root_name]['inertia']['izz'])      

        add_child_frame(root_name, root_frame, link_dict, joints, prefix)


    # add damping to joint if specified
    try:
        damping = system.get_force('system-damping') #find damping if previously created
    except KeyError:
        damping = trep.forces.Damping(system, 0.0, name='system-damping') #create damping class if not

    for joint in joints:
        jdyn=joint.find('dynamics')
        if jdyn is not None:
            jdamp=jdyn.get('damping')
            if jdamp is not None and system.get_config(prefix + joint.get('name')).kinematic is not True:
                damping.set_damping_coefficient(prefix + joint.get('name'), float(jdamp))
    
    return system
