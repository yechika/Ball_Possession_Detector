from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self, n_clusters=2):
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.team_colors = {0: "Team A", 1: "Team B"}  # Default team colors

    def get_player_team(self, frame, bbox, player_id):
        x, y, w, h = bbox
        try:
            top_half_image = frame[y:y + h // 2, x:x + w]
            if top_half_image.size == 0:
                raise ValueError(f"Player {player_id}: Extracted top_half_image is empty. Check bbox dimensions.")

            image_2d = top_half_image.reshape(-1, 3)
            self.kmeans.fit(image_2d)

            # Assume team is assigned based on the first cluster
            team_label = self.kmeans.labels_[0]
            return team_label
        except Exception as e:
            print(f"Error in get_player_team for player {player_id}: {e}")
            return -1  # Return -1 to indicate unknown team

    def assign_team_color(self, frame, bbox):
        try:
            x, y, w, h = bbox
            top_half_image = frame[y:y + h // 2, x:x + w]
            if top_half_image.size == 0:
                raise ValueError("Extracted top_half_image is empty. Ensure bbox is valid.")

            image_2d = top_half_image.reshape(-1, 3)
            self.kmeans.fit(image_2d)
            self.team_colors = {0: "Team A", 1: "Team B"}  # Update team colors if clustering succeeds
        except Exception as e:
            print(f"Error in assign_team_color: {e}")
