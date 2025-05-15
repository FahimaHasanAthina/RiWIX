import torch
from torchvision import transforms
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from PIL import Image
from swin_unet import swin_Unet
import numpy as np
import matplotlib.pyplot as plt  
import torchvision
import numpy as np
from skimage.morphology import skeletonize, medial_axis, thin, remove_small_objects, binary_erosion
from skimage.util import invert
import cv2
from scipy.ndimage import median_filter, distance_transform_edt
import networkx as nx
from skimage.measure import label, regionprops
from scipy.spatial import cKDTree


def PrepareInput(path):
    data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]) 
                         ])
    
    input_image = Image.open(path).convert('RGB')
    input_image = data_transforms(input_image)
    return input_image

def predict(input_image, model):
    model.eval()
    prediction = torch.sigmoid(model(input_image.unsqueeze(0)))
    prediction = (prediction > 0.5).float()
    prediction = prediction.squeeze(1)

    return prediction

def Plots(image, ground_truth, segmentation):
    fig, axes = plt.subplots(1, 3)

    # Plot the input image
    axes[0].imshow(np.transpose(image, (1, 2, 0)))
    axes[0].set_title('input image')
    axes[0].axis('off')  # Hide axis

    # Plot the ground truth
    axes[1].imshow(np.transpose(ground_truth, (1, 2, 0)), cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')  # Hide axis

    # Plot the prediction
    axes[2].imshow(np.transpose(segmentation, (1,2,0)), cmap='gray')
    axes[2].set_title('Segmented river')
    axes[2].axis('off')  # Hide axis

    # Show both figures
    plt.tight_layout() 
    plt.show()

def Centerline(river):

    # Step 1: Label connected river components
    labeled_rivers = label(river)
    
    # Step 2: Prepare an empty output mask
    final_centerline = np.zeros_like(river, dtype=np.uint8)

    # Step 3: Process each river separately
    for region in regionprops(labeled_rivers):
        minr, minc, maxr, maxc = region.bbox  # Bounding box
        river_mask = labeled_rivers[minr:maxr, minc:maxc] == region.label

        # Step 4: Skeletonize the individual river
        skeleton = skeletonize(river_mask)

        # Step 5: Convert skeleton to a graph
        graph = nx.Graph()
        rows, cols = np.where(skeleton)
        
        for (r, c) in zip(rows, cols):
            graph.add_node((r, c))
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in graph.nodes:
                        graph.add_edge((r, c), (nr, nc))

        # Step 6: Find the longest path in this river
        if graph.nodes:
            start_node = max(graph.nodes, key=lambda n: sum(1 for _ in nx.bfs_edges(graph, n)))
            lengths, paths = nx.single_source_dijkstra(graph, source=start_node)
            end_node = max(lengths, key=lengths.get)
            longest_path = paths[end_node]

            # Step 7: Add this centerline to the final image
            for r, c in longest_path:
                final_centerline[minr + r, minc + c] = 1
    return final_centerline
   

   

    #Skeletonize the river
    # skeleton = skeletonize(river)
    # skeleton = thin(skeleton)

    # # Convert skeleton to a graph
    # graph = nx.Graph()
    # rows, cols = np.where(skeleton > 0)
    
    # for (r, c) in zip(rows, cols):
    #     graph.add_node((r, c))
    #     for dr in [-1, 0, 1]:
    #         for dc in [-1, 0, 1]:
    #             if dr == 0 and dc == 0:
    #                 continue
    #             nr, nc = r + dr, c + dc
    #             if (nr, nc) in graph.nodes:
    #                 graph.add_edge((r, c), (nr, nc))

    # # Find the longest path (main centerline)
    # longest_path = []
    # if graph.nodes:
    #     start_node = max(graph.nodes, key=lambda n: sum(1 for _ in nx.bfs_edges(graph, n)))
    #     lengths, paths = nx.single_source_dijkstra(graph, source=start_node)
    #     end_node = max(lengths, key=lengths.get)
    #     longest_path = paths[end_node]

    # # Create a clean centerline mask
    # clean_skeleton = np.zeros_like(skeleton, dtype=np.uint8)
    # for r, c in longest_path:
    #     clean_skeleton[r, c] = 1

  

def get_boundary(river):
    """Extracts the boundary points of the river mask."""
    river = river.astype(bool)
    eroded = binary_erosion(river).astype(bool)
    boundary = river ^ eroded  # XOR to get only the boundary pixels
    return np.column_stack(np.where(boundary))  # Get (row, col) coordinates

def compute_river_width(river, centerline, boundary_points):
    # Compute distance transform (distance from river pixels to nearest boundary)
    distance_transform = distance_transform_edt(river)

    # Get centerline points
    centerline_points = np.column_stack(np.where(centerline))  # (row, col) format

    # Sample points at regular intervals
    sampled_points = centerline_points[::50]

    kdtree = cKDTree(boundary_points)

    # Compute width at each sampled centerline point
    widths = []
    closest_boundary_pts = []

    for r, c in sampled_points:
        width = 2 * distance_transform[r, c] * 0.3  # Width = 2 * distance from centerline to boundary
        widths.append(width)

        # Find the nearest boundary point
        _, nearest_idx = kdtree.query((r, c))
        closest_boundary_pts.append(boundary_points[nearest_idx])

    return sampled_points, widths, closest_boundary_pts



input_image_path = "c:\\Users\\fa578s\\Desktop\\preprocessed\\GLH\\test\\img\\156.jpg"
ground_truth_path = "c:\\Users\\fa578s\\Desktop\\preprocessed\\GLH\\test\\label\\156.png"
ground_truth = Image.open(ground_truth_path)
ground_truth = transforms.ToTensor()(ground_truth)
plot_input = Image.open(input_image_path)
plot_input = transforms.ToTensor()(plot_input)
input_image = PrepareInput(input_image_path)
# load model
model = swin_Unet(img_size=896, num_classes=1)
checkpoint = torch.load('my_checkpoint.pth.tar')
model.load_state_dict(checkpoint["state_dict"])
# segment the river
segmented_river = predict(input_image, model)
river = median_filter(segmented_river.detach().cpu().numpy().squeeze(0), size=20)

# Plots(plot_input, ground_truth, segmented_river)

centerline = Centerline(river)
boundary_points = get_boundary(river)
point, widths, closest_boundary_points = compute_river_width(river, centerline, boundary_points)


plt.imshow(river, cmap='gray')
plt.contour(centerline, colors='r')
plt.scatter(boundary_points[:, 1], boundary_points[:, 0], c="blue", s=10, label="Boundary Points")
plt.scatter(point[:, 1], point[:, 0], c=widths, cmap="jet", s=50, edgecolors="black")  # Color by width
# Draw width lines
for (cx, cy), (bx, by) in zip(point, closest_boundary_points):
    plt.plot([cy, by], [cx, bx], "green", linewidth=1) 
plt.colorbar(label="River Width (m))")
plt.title("River Width at Every 50 Points")
plt.xlabel("X (columns)")
plt.ylabel("Y (rows)")
# plt.gca().invert_yaxis()  # Invert y-axis for correct orientation
plt.show()
# plt.scatter(boundary_points[:, 1], boundary_points[:, 0], c="blue", s=10, label="Boundary Points")
    
# plt.tight_layout() 
# plt.show()
