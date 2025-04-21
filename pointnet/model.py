import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.enabled = False

class STNKd(nn.Module):
    # T-Net a.k.a. Spatial Transformer Network
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        """
        Input: [B,k,N]
        Output: [B,k,k]
        """
        B = x.shape[0]
        device = x.device
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]

        x = self.fc(x)
        
        # Followed the original implementation to initialize a matrix as I.
        identity = (
            Variable(torch.eye(self.k, dtype=torch.float))
            .reshape(1, self.k * self.k)
            .expand(B, -1)
            .to(device)
        )
        x = x + identity
        x = x.reshape(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    """
    Corresponds to the part that extracts max-pooled features.
    """
    def __init__(
        self,
        input_transform: bool = False,
        feature_transform: bool = False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if self.input_transform:
            self.stn3 = STNKd(k=3)
        if self.feature_transform:
            self.stn64 = STNKd(k=64)

        # point-wise mlp
        # TODO : Implement point-wise mlp model based on PointNet Architecture.
        self.mlp1 = nn.Sequential(nn.Conv1d(3, 64, 1) ,nn.BatchNorm1d(64),nn.ReLU(),\
                                  nn.Conv1d(64, 64, 1) ,nn.BatchNorm1d(64), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Conv1d(64, 128, 1) ,nn.BatchNorm1d(128),nn.ReLU(),\
                                nn.Conv1d(128, 1024, 1) ,nn.BatchNorm1d(1024),nn.ReLU())

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - Global feature: [B,1024]
            - ...
        """
        # TODO : Implement forward function.
        pointcloud = pointcloud.transpose(1, 2)
        if self.input_transform:
            feature1 = self.stn3(pointcloud)
            x = torch.bmm(feature1, pointcloud)

        feature2 = self.mlp1(x)

        if self.feature_transform:
            trans_feat = self.stn64(feature2)
            feature2 = torch.bmm(trans_feat, feature2)

        feature4 = self.mlp2(feature2)

        result = torch.max(feature4, 2)[0]
        return result


class PointNetCls(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        super().__init__()
        self.num_classes = num_classes
        
        # extracts max-pooled features
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)
        
        # returns the final logits from the max-pooled features.
        # TODO : Implement MLP that takes global feature as an input and return logits.
        self.cls_layer = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),\
                                       nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),\
                                        nn.Linear(256, self.num_classes), nn.BatchNorm1d(self.num_classes))

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - logits [B,num_classes]
            - ...   
        """
        # TODO : Implement forward function.
        # pointcloud = pointcloud.transpose(1, 2)
        pointcloud = self.pointnet_feat(pointcloud)
        return self.cls_layer(pointcloud)


class PointNetPartSeg(nn.Module):
    def __init__(self, m=50):
        super().__init__()

        # returns the logits for m part labels each point (m = # of parts = 50).
        # TODO: Implement part segmentation model based on PointNet Architecture.

        self.stn3 = STNKd(k=3)
        self.stn64 = STNKd(k=64)
        self.mlp1 = nn.Sequential(nn.Conv1d(3, 64, 1) ,nn.BatchNorm1d(64),nn.ReLU(),\
                                  nn.Conv1d(64, 64, 1) ,nn.BatchNorm1d(64), nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Conv1d(64, 128, 1) ,nn.BatchNorm1d(128),nn.ReLU(),\
                                nn.Conv1d(128, 1024, 1) ,nn.BatchNorm1d(1024))
        self.output1 = nn.Sequential(nn.Conv1d(1088, 512, 1) ,nn.BatchNorm1d(512),nn.ReLU(),\
                                    nn.Conv1d(512, 256, 1) ,nn.BatchNorm1d(256),nn.ReLU(),\
                                    nn.Conv1d(256, 128, 1) ,nn.BatchNorm1d(128),nn.ReLU())
        self.output2 = nn.Sequential(nn.Conv1d(128, 128, 1) ,nn.BatchNorm1d(128),nn.ReLU(),\
                                    nn.Conv1d(128, m, 1))

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - logits: [B,50,N] | 50: # of point labels
            - ...
        """
        # TODO: Implement forward function.
        B, N, _ = pointcloud.shape
        pointcloud = pointcloud.transpose(1, 2)
        feature1 = self.stn3(pointcloud)
        x = torch.bmm(feature1, pointcloud)

        feature2 = self.mlp1(x)

        trans_feat = self.stn64(feature2)
        local_feature = torch.bmm(trans_feat, feature2)

        feature4 = self.mlp2(local_feature)

        global_feature = torch.max(feature4, 2)[0]
        gloabl_feature = global_feature.unsqueeze(2).repeat(1, 1, N)

        feature = torch.cat((local_feature, gloabl_feature), dim=1)

        pnt_feat = self.output1(feature)
        result = self.output2(pnt_feat)

        return result


class PointNetAutoEncoder(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.pointnet_feat = PointNetFeat(input_transform=True, feature_transform=True)

        # Decoder is just a simple MLP that outputs N x 3 (x,y,z) coordinates.
        # TODO : Implement decoder.
        self.num_points = num_points
        self.ae_layer1 = nn.Sequential(nn.Linear(1024, self.num_points // 4), nn.BatchNorm1d(self.num_points // 4),nn.ReLU())
        self.ae_layer2 = nn.Sequential(nn.Linear(self.num_points // 4, self.num_points // 2), nn.BatchNorm1d(self.num_points // 2),nn.ReLU())
        self.ae_layer3 = nn.Sequential(nn.Linear(self.num_points // 2, self.num_points), nn.Dropout(0.1), nn.BatchNorm1d(self.num_points),nn.ReLU(),\
                                    nn.Linear(self.num_points, self.num_points * 3, 1))                                    
                                    

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - pointcloud [B,N,3]
            - ...
        """
        # TODO : Implement forward function.
        B, N, _ = pointcloud.shape
        logits = self.pointnet_feat(pointcloud) # [B, N]
        result = self.ae_layer1(logits)
        result = self.ae_layer2(result)
        result = self.ae_layer3(result)
        return result.reshape(B, self.num_points, 3)

def get_orthogonal_loss(feat_trans, reg_weight=1e-3):
    """
    a regularization loss that enforces a transformation matrix to be a rotation matrix.
    Property of rotation matrix A: A*A^T = I
    """
    if feat_trans is None:
        return 0

    B, K = feat_trans.shape[:2]
    device = feat_trans.device

    identity = torch.eye(K).to(device)[None].expand(B, -1, -1)
    mat_square = torch.bmm(feat_trans, feat_trans.transpose(1, 2))

    mat_diff = (identity - mat_square).reshape(B, -1)

    return reg_weight * mat_diff.norm(dim=1).mean()
