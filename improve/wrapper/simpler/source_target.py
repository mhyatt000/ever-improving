



    @property
    def source_obj_pose(self):
        """Get the center of mass (COM) pose."""
        return self.episode_source_obj.pose.transform(
            self.episode_source_obj.cmass_local_pose
        )

    @property
    def target_obj_pose(self):
        """Get the center of mass (COM) pose."""
        return self.episode_target_obj.pose.transform(
            self.episode_target_obj.cmass_local_pose
        )

