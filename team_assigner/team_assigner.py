
from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self):
        self.team_colours = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        # Reshape the image into 2d array
        image_2d = image.reshape(-1, 3)

        # perform k-means clustering with 2 clusters
        # k-means++ helps to get better clusters fasters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_colour(self, frame, bbox):
        cropped_image = frame[
            int(bbox[1]):int(bbox[3]),
            int(bbox[0]):int(bbox[2])]

        top_half_image = cropped_image[0: int(cropped_image.shape[0] / 2), :]

        # Get the clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels
        labels = kmeans.labels_

        # reshape the labels back to the original image shape
        clustered_image = labels.reshape(
            top_half_image.shape[0], top_half_image.shape[1])

        # Get the Player Cluster
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1],
                           clustered_image[-1, 0], clustered_image[-1, -1]]

        non_player_cluster = max(set(corner_clusters),
                                 key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_colour = kmeans.cluster_centers_[player_cluster]

        return player_colour

    def assign_team_color(self, frame, player_detections):

        player_color = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_colour = self.get_player_colour(frame, bbox)
            player_color.append(player_colour)

        # We now want to divide the player colours into 2 teams. We can do this by performing k-means clustering with 2 clusters on the player colours.
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_color)

        self.kmeans = kmeans

        self.team_colours[1] = kmeans.cluster_centers_[0]
        self.team_colours[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_colour = self.get_player_colour(frame, player_bbox)

        team_id = self.kmeans.predict(player_colour.reshape(1, -1))[0]
        team_id += 1

        # Temp hardcode goalkeeper ids to team x as our model is not trained enough to detect goalkeepers
        # Read player_team_dict player 7 team
        player_7_team = self.player_team_dict.get(7, None)
        other_team = 2 if player_7_team == 1 else 1
        if player_7_team is not None:
            if player_id in [68, 72, 76]:
                team_id = player_7_team

        if player_id in [147, 156]:
            team_id = other_team

        self.player_team_dict[player_id] = team_id

        return team_id
