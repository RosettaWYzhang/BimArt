import numpy as np
from pysdf import SDF
import trimesh

class APDMetric():
    ''' This metric is used for both diversity and multimodality
    For multimodality, num_sample equals to 5
    For diversity, num_sample equals to 5 * all_test_sequences
    '''
    def __init__(self):
        self.apd = 0
        self.total = 0

    def update(self, pred_list):
        ''' 
        pred_list: np array of dim [num_sample, frame_number, joint_number, 3] 
        # here joint_number is all mano vertices
        '''
        assert pred_list.ndim == 4
        if pred_list.shape[0] < 10:
            self.apd += self.compute_apd_small(pred_list)
        else:
            self.apd += self.compute_apd_large(pred_list)
        self.total += 1 # always batch-wise

    def compute(self):
        if self.total == 0:
            return None
        return float('%.5g' % (self.apd / self.total * 100))
    
    def compute_apd_large(self, positions):
        """
        Compute the Average Pairwise Distance (APD) for a given set of joint positions.
        Memory-efficient, but slower
        
        Parameters:
        positions (np.ndarray): Array of shape [num_sample, frame_number, joint_number, 3]
        
        Returns:
        float: The APD metric as a single scalar value
        """
        num_samples, frame_number, joint_number, _ = positions.shape
        total_distance = 0
        count = 0
        
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                # Compute the MPJPE for the pair (i, j)
                pairwise_distance = np.mean(np.linalg.norm(positions[i] - positions[j], axis=-1))
                total_distance += pairwise_distance
                count += 1
        
        apd = total_distance / count if count > 0 else 0
        return apd

    def compute_apd_small(self, samples):
        """
        Compute the average pairwise Mean Per Joint Position Error (MPJPE) for all samples.
        Faster, but more memory-intensive.

        Parameters:
        samples (numpy.ndarray): Joint positions of shape (num_sample, frame_number, joint_number, 3).

        Returns:
        float: The average pairwise MPJPE across all samples.
        """
        num_sample, frame_number, joint_number, _ = samples.shape
        
        # Reshape samples for broadcasting
        samples_expanded_1 = samples[:, np.newaxis, :, :, :]  # Shape (num_sample, 1, frame_number, joint_number, 3)
        samples_expanded_2 = samples[np.newaxis, :, :, :, :]  # Shape (1, num_sample, frame_number, joint_number, 3)
        
        # Compute the Euclidean distance for each pair of samples
        errors = np.linalg.norm(samples_expanded_1 - samples_expanded_2, axis=-1)  # Shape (num_sample, num_sample, frame_number, joint_number)

        # Compute the mean error over all joints and frames for each pair of samples
        mpjpe_matrix = np.mean(errors, axis=(2, 3))  # Shape (num_sample, num_sample)
        
        # Compute the average of the upper triangle of the pairwise distance matrix (excluding the diagonal)
        upper_triangle_indices = np.triu_indices(num_sample, k=1)
        average_pairwise_mpjpe = np.mean(mpjpe_matrix[upper_triangle_indices])
        
        return average_pairwise_mpjpe
    
    
class JitterMetric:
    def __init__(self):
        self.jitter_sum = 0
        self.total_samples = 0

    def update(self, pred_positions):
        ''' 
        Update the jitter metric with a new batch of predicted positions.
        
        Parameters:
        pred_positions (np.ndarray): Predicted positions of shape [frame_number, joint_number, 3]
        '''
        assert pred_positions.ndim == 3, "pred_positions must have 3 dimensions"
        self.jitter_sum += self.compute_jitter(pred_positions)
        self.total_samples += 1

    def compute(self):
        '''Compute the average jitter.'''
        if self.total_samples == 0:
            return None
        return float('%.5g' % (self.jitter_sum / self.total_samples * 100))
    
    def compute_jitter(self, pred_positions):
        ''' 
        Compute the jitter for the predicted positions per joint.
        
        Parameters:
        pred_positions (np.ndarray): Predicted positions of shape [frame_number, joint_number, 3]
        
        Returns:
        float: The mean jitter across all joints.
        '''
        # Compute the acceleration for each frame and joint
        acceleration = np.diff(np.diff(pred_positions, axis=0), axis=0)  # Shape: [frame_number-2, joint_number, 3]
        
        # Compute the Euclidean norm of acceleration vectors for each joint across all frames
        joint_jitter = np.linalg.norm(acceleration, axis=-1)  # Shape: [frame_number-2, joint_number]
        
        # Compute the mean jitter across all frames and joints
        overall_mean_jitter = np.mean(joint_jitter)  # Shape: scalar
        
        return overall_mean_jitter
    

class PenetrationDepthMetric(): 
    '''To make it consistent with CAMS
    still a percentage error
    '''
    def __init__(self, thres=0.005):
        self.pn = 0
        self.total = 0
        self.thres = thres

    def update(self, hand_verts, obj_verts, obj_faces):
        
        
        self.pn += self.get_frame_pen_count(hand_verts, obj_verts, obj_faces)
        #self.total += 1 # always batch-wise, if doing hand verts average
        self.total += hand_verts.shape[0] # framewise

    def compute(self):
        if self.total == 0:
            return None
        pen_percent = self.pn / self.total * 100
        # 5df        
        return float('%.5g' % pen_percent)
    
    def get_frame_hand_pen_count(self, hand_verts, obj_verts, obj_faces):
        ''' get a frame-percentage of hands verts which are at least 5mm inside the object
        '''
        max_u = 0
        for i in range(obj_verts.shape[0]):
            sdf = SDF(obj_verts[i], obj_faces)
            u = np.where(sdf(hand_verts[i]) > self.thres)[0]
            # check what np.max return
            max_u = max_u + u.shape[0]
        return max_u / hand_verts.shape[0] / hand_verts.shape[1]

    def get_pen_depth(self, hand_verts, obj_verts, obj_faces):
        ''' get a frame-percentage of hands verts which are at least 5mm inside the object
        '''
        pen_count = 0
        pen_depth = 0
        for i in range(obj_verts.shape[0]):
            sdf = SDF(obj_verts[i], obj_faces)
            sdf_check = sdf(hand_verts[i]) 
            u = np.where(sdf_check > self.thres)[0]
            # check what np.max return
            pen_depth += np.sum(sdf_check[u])
            pen_count += u.shape[0]
        return pen_count, pen_depth
    
    def get_frame_pen_count(self, hand_verts, obj_verts, obj_faces):
        '''This is activately used!'''
        ''' get a frame-percentage of hands verts which are at least 5mm inside the object
        '''
        max_u = 0
        for i in range(obj_verts.shape[0]):
            sdf = SDF(obj_verts[i], obj_faces)
            u = np.max(sdf(hand_verts[i])) > self.thres # positive means inside of the object
            # check what np.max return
            # import pdb; pdb.set_trace()
            # ps.init()
            # ps.register_point_cloud("obj", obj_verts[i])
            # ps.register_point_cloud("hand", hand_verts[i])
            # ps.show()
            if u:
                max_u += 1
        #print("sequence avg is ", max_u / obj_verts.shape[0])
        return max_u 

      
class ContactPercentageMetric(): 
    '''Percentage of frames which are in contact with the objects.
    A frame of object is considered in contact when a single hand vertex (out of two hands) satisfies a distance threshold.
    '''

    def __init__(self):
        self.con = 0
        self.total = 0 # total frame number

    def update(self, hand_verts, obj_verts, obj_faces):
        frame_count, total_count = self.get_num_frames_in_contact(hand_verts, obj_verts, obj_faces)
        self.con += frame_count
        self.total += total_count 

    def compute(self):
        if self.total == 0:
            return None
        return float('%.5g' % (self.con / self.total * 100))
    
    def get_num_frames_in_contact(self, hand_verts, obj_verts, obj_faces):
        '''hands_verts, obj_verts are in global coordinates
        '''
        frame_count = 0
        total_count = 0
        for i in range(obj_verts.shape[0] - 1):
            if np.max(np.abs(obj_verts[i+1] - obj_verts[i])) > 1e-5: # object is moving, account for numerical error 
                # for each frame 
                sdf = SDF(obj_verts[i], obj_faces)
                u = np.where(sdf(hand_verts[i]) > -0.005)[0] # negative means away from the surface
                if u.shape[0] > 0:
                    frame_count += 1
                total_count += 1
        return frame_count, total_count
    
class ArtiContactPercentageMetric(): 
    '''Percentage of frames which fingers are in contact with the articulated part (if the articulated part is moving)
    A frame of object is considered in contact when a single hand vertex (out of two hands) satisfies a distance threshold.
    '''

    def __init__(self):
        self.con = 0
        self.total = 0 # total frame number
        self.top_ind = 0 # articulation part index

    def update(self, hand_verts, obj_verts, obj_faces, obj_parts, obj_states):
        total_frame, contact_frame = self.get_num_frames_in_contact_with_articulated_part(
            hand_verts, obj_verts, obj_faces, obj_parts, obj_states)
        self.con += contact_frame
        self.total += total_frame

    def compute(self):
        if self.total == 0:
            return None
        return float('%.5g' % (self.con / self.total * 100))
    
    def filter_and_reconstruct_mesh(self, verts, faces, segment_mask):
        """
        Filters vertices and faces based on a segmentation mask, then reconstructs the mesh to ensure it is closed.

        Args:
            verts (list of tuples): List of vertices.
            faces (list of tuples): List of faces, each face is a tuple of vertex indices.
            segment_mask (list of int): List of 0s and 1s indicating if the vertex is in the top (1) or bottom (0) part.

        Returns:
            new_verts (list of tuples): New list of vertices for the top part.
            new_faces (list of tuples): New list of faces for the top part, including additional faces to close the mesh.
        """
        # Step 1: Filter vertices
        vert_map = {i: new_idx for new_idx, i in enumerate(v for v, mask in enumerate(segment_mask) if mask == self.top_ind)}
        new_verts = [verts[i] for i in vert_map.keys()]

        # Step 2: Filter and reindex faces
        new_faces = []
        boundary_edges = set()

        for face in faces:
            # Check if all vertices in this face belong to the top part
            face_in_top = [v for v in face if segment_mask[v] == self.top_ind]
            
            if len(face_in_top) == 3:
                # All vertices belong to the top part
                new_faces.append(tuple(vert_map[v] for v in face_in_top))
            elif len(face_in_top) == 2:
                # Boundary edge, potential gap
                v1, v2 = tuple(vert_map[v] for v in face_in_top)
                boundary_edges.add((min(v1, v2), max(v1, v2)))

        return new_verts, new_faces
    
    def get_num_frames_in_contact_with_articulated_part(self, hand_verts, obj_verts, obj_faces, obj_parts, obj_states):
        '''
        Args:
           obj_verts are canonical vertices of the object
           obj_parts is used to get the articulated part of the object
           obj_states: used to test if there is a change in articulation by calculating the difference of articulation angle
                       dimension: [1, ph, 7]
        '''
        contact_frame_count = 0
        total_count = 0
        obj_states = obj_states.squeeze(0) # [ph, 7]
        for i in range(obj_verts.shape[0]-1): # do not consider the last frame
            # for each frame, only increase count if it is moving
            if np.sum(np.abs(obj_states[i+1, 0] - obj_states[i, 0])) > 0.001:  
                total_count += 1
                moving_verts, moving_faces = self.filter_and_reconstruct_mesh(obj_verts[i], obj_faces, obj_parts)
                test_trimesh = trimesh.Trimesh(vertices=moving_verts, faces=moving_faces)
                sdf = SDF(test_trimesh.vertices, test_trimesh.faces) 
                u = np.where(sdf(hand_verts[i]) > -0.005)[0] # negative means away from the surface
                if u.shape[0] > 0:
                    contact_frame_count += 1
        return total_count, contact_frame_count   
