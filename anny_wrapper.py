"""
anny_wrapper.py

Wrapper for the ANNY body model.
Makes ANNY usable in the Body Model Visualizer next to SMPL, SUPR, STAR, etc.

ANNY has a different API:
- uses bone matrices (4x4) instead of axis-angle tensors
- uses phenotype params (age, height, ...) instead of betas
- uses Z-up coordinate system instead of Y-up

This wrapper translates between the two so main.py doesn't need to care.
"""

#imports
import torch
import numpy as np
import anny
from scipy.spatial.transform import Rotation as R

class AnnyOutput:
    """
    Fake SMPL-style output container.
    Has .vertices and .joints so main.py can use it like SMPL output.
    """
    def __init__(self, vertices, joints):
        self.vertices = vertices
        self.joints = joints

class AnnyWrapper:
    """
    Wraps the ANNY body model with a SMPL-style API.
    
    Lets main.py call ANNY just like SMPL/SUPR/STAR:
        model(betas=..., pose=..., trans=...)
    
    Translates between:
    - SMPL-style pose tensor [N*3] <-> ANNY bone matrix dict
    - ANNY bone_poses [4x4] <-> SMPL-style joints [3]
    - ANNY Z-up <-> visualizer Y-up
    """
    
    def __init__(self, gender='neutral', rig='default'):
        # Load the ANNY model itself
        self.anny_model = anny.create_fullbody_model(
            rig=rig,
            local_changes=True,
            all_phenotypes=True,
            triangulate_faces=True,
        )
        
        # Remember which gender/rig we use
        self.gender = gender
        self.rig = rig
        
        # SMPL-compatible attributes (main.py uses these)
        # faces need to be numpy for open3d
        self.faces = self.anny_model.faces.detach().numpy()
        
        # Bone info (for GUI later)
        self.bone_labels = self.anny_model.bone_labels
        self.n_bones = len(self.bone_labels)
        
        # Storage for ANNY-specific features (set via setters later)
        self.phenotypes = {}
        self.local_changes = {}

    def _z_up_to_y_up(self, points):
        """
        Convert Z-up coordinates (ANNY/Blender) to Y-up (SMPL/visualizer).
        Rotation by 90 degrees around the X-axis.
        
        Works for any shape ending in 3: [3], [N, 3], [B, N, 3], ...
        """
        result = points.clone()
        result[..., 1] = points[..., 2]    # new Y = old Z
        result[..., 2] = -points[..., 1]   # new Z = -old Y
        return result

    def _axis_angle_to_matrix(self, axis_angle):
        """
        Convert an axis-angle vector to a 3x3 rotation matrix (Rodrigues formula)
        
        Input:  tensor [3] (axis-angle: direction = axis, magnitude = angle in radians)
        Output: tensor [3, 3] (rotation matrix)
        """
        # Use scipy to do the Rodrigues conversion
        axis_angle_np = axis_angle.detach().numpy()
        rot_matrix = R.from_rotvec(axis_angle_np).as_matrix()
        return torch.from_numpy(rot_matrix).double()

    def set_phenotypes(self, phenotypes_dict):
        """
        Set ANNY phenotype parameters from outside.
        Called by main.py to push GUI values to the wrapper.
        """
        self.phenotypes = phenotypes_dict
    
    def set_local_changes(self, local_changes_dict):
        """
        Set ANNY local change parameters from outside.
        Called by main.py to push GUI values to the wrapper.
        """
        self.local_changes = local_changes_dict
    
    def set_rig(self, rig_name):
        """
        Change the rig (default, mixamo, cmu_mb, etc.).
        Reloads the ANNY model with the new rig.
        """
        if rig_name != self.rig:
            self.rig = rig_name
            self.anny_model = anny.create_fullbody_model(
                rig=rig_name,
                local_changes=True,
                all_phenotypes=True,
                triangulate_faces=True,
            )
            # Update bone info because rigs can have different bones!
            self.bone_labels = self.anny_model.bone_labels
            self.n_bones = len(self.bone_labels)
            self.faces = self.anny_model.faces.detach().numpy()

    def _convert_output(self, anny_output):
        """
        Convert ANNY's output dict into our AnnyOutput container
        
        ANNY returns a dict with:
            'vertices':    Tensor [B, V, 3]    - mesh vertices (Z-up)
            'bone_poses':  Tensor [B, N, 4, 4] - bone matrices (Z-up)
            (and more, but we don't need the rest)
        
        We need:
            output.vertices:   Tensor [B, V, 3]   - same shape, Y-up
            output.joints:     Tensor [B, N, 3]   - only positions, Y-up
        """
        # Get the parts we need from ANNY's output dict
        verts = anny_output['vertices']         # [B, V, 3]
        bone_poses = anny_output['bone_poses']  # [B, N, 4, 4]
        
        # Extract joint positions from bone matrices (the shortcut!)
        # bone_poses[..., :3, 3] = translation part of each 4x4 matrix
        joints = bone_poses[..., :3, 3]          # [B, N, 3]
        
        # Convert from ANNY's Z-up to visualizer's Y-up
        verts = self._z_up_to_y_up(verts)
        joints = self._z_up_to_y_up(joints)
        
        # Make sure types are float32 (not float64) for compatibility
        verts = verts.float()
        joints = joints.float()
        
        # Pack into our SMPL-style container
        return AnnyOutput(vertices=verts, joints=joints)

    def _build_rig_pose(self, pose, trans):
        """
        Convert SMPL-style pose tensor + translation into ANNY's bone dictionary
        
        SMPL-style input:
            pose:  Tensor [1, N*3]   axis-angle for each bone (flat)
            trans: Tensor [1, 3]     translation for the root
        
        ANNY-style output:
            dict {bone_name: 4x4 matrix} with shape [1, 4, 4] per bone
            - root bone: gets translation + rotation
            - other bones: only rotation (translation = 0)
        """
        rig_pose = {}
        
        # Start with identity matrices for all bones
        # FIX: ANNY expects shape [batch_size, 4, 4], not [4, 4]
        # → .unsqueeze(0) adds the batch dimension: [4,4] becomes [1,4,4]
        for bone_name in self.bone_labels:
            rig_pose[bone_name] = torch.eye(4, dtype=torch.float64).unsqueeze(0)
        
        # Add translation to the root bone (only if provided)
        if trans is not None:
            root_name = self.bone_labels[0]
            # FIX: matrix is now [1, 4, 4] - need [0, :3, 3] instead of [:3, 3]
            #      to access through the batch dimension
            rig_pose[root_name][0, :3, 3] = trans.flatten()[:3]
        
        # Add rotations to all bones (only if pose provided)
        if pose is not None:
            pose_per_bone = pose.reshape(-1, 3)
            
            for i, bone_name in enumerate(self.bone_labels):
                axis_angle = pose_per_bone[i]
                rot_matrix = self._axis_angle_to_matrix(axis_angle)
                # FIX: matrix is now [1, 4, 4] - need [0, :3, :3] instead of [:3, :3]
                rig_pose[bone_name][0, :3, :3] = rot_matrix
        
        return rig_pose

    def __call__(self, betas=None, pose=None, trans=None, **kwargs):
        """
        Main entry point - called by main.py like model(...).
        
        Translates SMPL-style inputs to ANNY format, runs ANNY,
        and converts the output back to SMPL-style.
        
        Args:
            betas:  ignored (ANNY uses phenotypes instead, set via setter)
            pose:   Tensor [1, N*3] - axis-angle rotations per bone
            trans:  Tensor [1, 3]   - translation for the root
            **kwargs: any extra arguments (ignored)
        
        Returns:
            AnnyOutput with .vertices [1, V, 3] and .joints [1, N, 3]
        """
        # 1. Convert SMPL-style pose to ANNY-style bone dict
        rig_pose = self._build_rig_pose(pose, trans)
        
        # 2. Call the actual ANNY model
        anny_output = self.anny_model(
            pose_parameters=rig_pose,
            phenotype_kwargs=self.phenotypes,
            local_changes_kwargs=self.local_changes,
        )
        
        # 3. Convert ANNY's output to SMPL-style
        return self._convert_output(anny_output)


