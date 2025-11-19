# Query Object Processing Pipeline

This document describes the implementation of the **Query Object Processing** phase from the paper, which extracts both geometric and visual features from a 3D object for 6-DoF pose estimation.

---

## Pipeline Architecture

```
Input: object.xyz (3D point cloud)
   ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 1: Geometric Feature Extraction                         │
│ Ψ(P_Q) → G_Q = {g_{Q,n}}_{n=1..N}                           │
└──────────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 2: Multi-view Rendering                                 │
│ Render {I_r}_{r=1..R} from R viewpoints with masks          │
└──────────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 3: Visual Feature Extraction                            │
│ Φ(I_r^Q) → V_r^Q for each view r                            │
└──────────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 4: Feature Back-projection                              │
│ Project V_r^Q to 3D points using camera parameters          │
└──────────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 5: Multi-view Aggregation                               │
│ v_{Q,n} = (Σ_{r=1..R} v_{Q,r,n}) / R                        │
└──────────────────────────────────────────────────────────────┘
   ↓
Output: G_Q [N × G] + V_Q [N × D]
```

---

## Mathematical Formulation

### Geometric Features

Given a query object point cloud $P_Q = \{p_n\}_{n=1}^N$ where $p_n \in \mathbb{R}^3$:

$$G_Q = \Psi(P_Q) = \{g_{Q,n}\}_{n=1}^N, \quad g_{Q,n} \in \mathbb{R}^G$$

Where:
- $\Psi$: Geometric encoder (GeDi)
- $G$: Geometric feature dimension (32 in our implementation)
- $N$: Number of points in the point cloud

**Implementation:** `1_gedi_process_object.py`

```python
# Load point cloud
pts = torch.tensor(np.asarray(pcd.points)).float()

# Compute geometric descriptors
desc = gedi.compute(pts=pts, pcd=pts)  # G_Q: [N, 32]
```

---

### Visual Features

#### 1. Multi-view Rendering

Render the textured 3D model from $R$ different viewpoints:

$$\{I_r\}_{r=1}^R, \quad I_r \in \mathbb{R}^{H \times W \times 4}$$

Where each $I_r$ contains RGB channels and a depth channel.

**Viewpoints used:** 
- Front, Right, Left, Back, Top, Bottom ($R = 6$)
- Resolution: $960 \times 720$ pixels

**Implementation:** `1_gedi_process_object.py`

```python
view_rotations = {
    "front":  [np.pi, 0, 0],
    "right":  [np.pi, -np.pi/2, 0],
    "left":   [np.pi, np.pi/2, 0],
    "back":   [np.pi, np.pi, 0],
    "top":    [np.pi/2, 0, 0],
    "bottom": [-np.pi/2, 0, 0],
}

for i, (name, rot) in enumerate(view_rotations.items()):
    capture_view_with_mask(pcd, rgb_path, mask_path, depth_path, rot)
```

#### 2. Segmentation Masks

Generate binary masks $M_r$ to isolate the object region:

$$I_r^Q = I_r \odot M_r$$

Where $\odot$ denotes element-wise multiplication (masking).

**Implementation:** `1_gedi_process_object.py`

```python
# Create mask from white background
rgb_array = np.array(rgb_img)
white_threshold = 250
is_white = np.all(rgb_array >= white_threshold, axis=-1)
mask = (~is_white).astype(np.uint8) * 255
```

#### 3. Dense Visual Feature Extraction

Process each masked view with the vision encoder $\Phi$ (DINOv2):

$$V_r^Q = \Phi(I_r^Q) = \{v_{r,k}\}_{k=1}^{K_r}$$

Where:
- $\Phi$: Vision encoder (DINOv2 ViT-L/14)
- $v_{r,k} \in \mathbb{R}^D$: Feature vector for pixel $k$
- $D$: Visual feature dimension (1024 for DINOv2-ViT-L)
- $K_r$: Number of masked pixels in view $r$

**Key parameters:**
- Input resolution: $518 \times 518$ (divisible by patch size 14)
- Patch size: $14 \times 14$
- Grid size: $37 \times 37 = 1369$ patches

**Implementation:** `demo_infer.py`

```python
# Preprocessing
IMG_SIZE = 518  # 518 = 37 * 14
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Extract features
output = model.get_intermediate_layers(x, n=1)[0]  # [1, 1370, 1024]
patch_tokens = output[:, 1:, :]  # Remove CLS token: [1, 1369, 1024]

# Reshape to spatial grid
grid_size_h = grid_size_w = IMG_SIZE // PATCH_SIZE  # 37
features_spatial = patch_tokens.reshape(1, grid_size_h, grid_size_w, 1024)
features_spatial = features_spatial.permute(0, 3, 1, 2)  # [1, 1024, 37, 37]

# Upsample to full resolution
features_upsampled = F.interpolate(
    features_spatial,
    size=(IMG_SIZE, IMG_SIZE),
    mode='bilinear',
    align_corners=False
)  # [1, 1024, 518, 518]

# Apply mask to extract object features
features_masked = features_upsampled * mask_binary
```

---

### Feature Back-projection

The paper describes the back-projection process:

> First, we compute the correspondences between the RGB pixels of $I_r^Q$ and the points of $P_Q$. We convert the depth channel of $I_r^Q$ into a viewpoint-dependent point cloud $P_{Q,r}$ using the renderer's camera intrinsic parameters, and employ nearest-neighbour search between the points of $P_{Q,r}$ and $P_Q$.

#### Camera Model

**Intrinsic Matrix:**

$$K = \begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}$$

Where for FOV = 60°:
- $f_x = f_y = \frac{W}{2 \tan(\text{FOV}/2)}$
- $c_x = W/2$, $c_y = H/2$

**Extrinsic Matrix:**

$$T = \begin{bmatrix}
R & t \\
0 & 1
\end{bmatrix} \in \mathbb{R}^{4 \times 4}$$

Where:
- $R \in \mathbb{R}^{3 \times 3}$: Rotation matrix
- $t \in \mathbb{R}^3$: Translation vector

**Implementation:** `2_backproject_features.py`

```python
def create_camera_intrinsic(width, height, fov=60):
    fx = fy = width / (2 * np.tan(np.radians(fov) / 2))
    cx = width / 2
    cy = height / 2
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
    return intrinsic
```

#### 3D to 2D Projection

For each 3D point $p_n = [x, y, z]^T \in P_Q$:

$$\begin{bmatrix} u \\ v \\ w \end{bmatrix} = K \cdot [R|t] \cdot \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}$$

$$\text{pixel}_x = u/w, \quad \text{pixel}_y = v/w$$

**Implementation:** `2_backproject_features.py`

```python
def project_points_to_image(points_3d, intrinsic, extrinsic, width, height):
    # Convert to homogeneous coordinates
    points_homo = np.hstack([points_3d, np.ones((N, 1))])
    
    # Transform to camera coordinates
    points_cam = (extrinsic @ points_homo.T).T
    x_cam, y_cam, z_cam = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    
    # Project to image plane
    K = intrinsic.intrinsic_matrix
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    
    x_img = (fx * x_cam / z_cam) + cx
    y_img = (fy * y_cam / z_cam) + cy
    
    return np.column_stack([x_img, y_img]), z_cam
```

#### Correspondence Matching

Use k-NN search to find correspondences between projected points and feature pixels:

$$\text{corr}(n, r) = \arg\min_{k} \|p_{n,r}^{2D} - f_k^{2D}\|_2$$

Where:
- $p_{n,r}^{2D}$: Projected 2D location of 3D point $n$ in view $r$
- $f_k^{2D}$: 2D location of feature pixel $k$
- Search radius: 10 pixels

**Implementation:** `2_backproject_features.py`

```python
# Build KD-tree for feature coordinates
feature_pixels = feature_coords[:, [1, 0]].astype(np.float32)  # (x, y)
tree = cKDTree(feature_pixels)

# For each projected point, find nearest feature
for idx in valid_indices:
    px, py = pixels_scaled[idx]
    distances, feat_idx = tree.query([px, py], k=1, distance_upper_bound=10)
    
    if not np.isinf(distances):
        point_features[idx] = features[feat_idx]  # Assign feature
        point_valid[idx] = True
```

---

### Multi-view Aggregation

The paper describes averaging features across views:

> Lastly, we aggregate multi-view features $\{V_r^Q\}_{r=1..R}$ into a single set $V_Q = \{v_{Q,n}\}_{n=1..N}$ by averaging the contribution of each viewpoint as:

$$v_{Q,n} = \frac{\sum_{r=1}^R v_{Q,r,n}}{R}$$

Where $v_{Q,r,n}$ is the visual feature of point $n$ from view $r$ (zero if not visible).

**Implementation:** `2_backproject_features.py`

```python
def aggregate_multiview_features(all_features, all_valid, method='mean'):
    # Stack features from all views
    features_stack = np.stack(all_features, axis=0)  # [R, N, D]
    valid_stack = np.stack(all_valid, axis=0)  # [R, N]
    
    # Average across views (only where valid)
    valid_expanded = valid_stack[:, :, np.newaxis]  # [R, N, 1]
    features_masked = features_stack * valid_expanded
    
    sum_features = features_masked.sum(axis=0)  # [N, D]
    count_features = valid_stack.sum(axis=0, keepdims=True).T  # [N, 1]
    count_features = np.maximum(count_features, 1)  # Avoid division by zero
    
    aggregated = sum_features / count_features
    
    return aggregated
```

---

## Output Files

### Geometric Features
- **File:** `Kinfu_Samurai1_light_gedi.npy`
- **Shape:** `[N, 32]` where N = 3993
- **Description:** GeDi geometric descriptors for each point

### Visual Features (Per View)
- **Files:** `dino_view_{00-05}_{viewname}.npy`
- **Content:**
  ```python
  {
      'features': np.array([K, 1024]),  # Feature vectors
      'coords': np.array([K, 2]),       # Pixel coordinates (y, x)
      'feature_dim': 1024,
      'num_masked_pixels': K
  }
  ```

### Aggregated Visual Features
- **File:** `Kinfu_Samurai1_light_visual_features.npy`
- **Content:**
  ```python
  {
      'features': np.array([N, 1024]),        # Aggregated features
      'num_views_per_point': np.array([N]),   # View count per point
      'points_3d': np.array([N, 3]),          # 3D coordinates
      'feature_dim': 1024
  }
  ```

### Visualizations
- **Files:** `visualizations/match_viz_{view}.png`
- **Shows:** Projected points, feature matches, statistics

---



## Visualization

Run the visualization script to inspect matching quality:

```bash
python visualize_matching.py processed/ Kinfu_Samurai1_light
```

**Generates:**
1. Per-view matching visualizations
2. Point cloud colored by view coverage
3. PCA visualization of visual features
4. Coverage statistics and histograms
