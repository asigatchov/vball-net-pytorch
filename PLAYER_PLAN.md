## Player Detector Plan

- Task: detect volleyball players as bounding boxes on a static-camera video.
- Input format: keep the current project convention, `9` grayscale frames stacked on channels.
- Main target: detect players, not spectators.
- Practical fallback: allow slightly generous detection and remove false positives by court-zone filtering.

## Accepted Decisions

- Model family: a new `VballNetPlayerGridV1` in the same style as `vballnet_grid_v1b.py`.
- Detector type: anchor-free multi-object grid detector, not a fixed `12`-slot regressor.
- Sequence target: predict boxes only for the central frame of the `9`-frame stack.
- Output stride: `8`, so `432x768 -> 54x96` prediction grid.
- Output channels per cell: `5`.
- Channel meaning: `objectness`, `dx`, `dy`, `w`, `h`.
- Box parameterization: center is assigned to one grid cell, `dx` and `dy` are offsets inside the cell, `w` and `h` are normalized by image width and height.
- Head activation: `Sigmoid` on all output channels to keep ONNX export and decoding simple.

## Architecture Decisions

- Backbone: depthwise separable convolution blocks, matching the style of existing grid models.
- Neck: a lightweight two-scale fusion block.
- Feature maps used by the neck: `stride=8` and `stride=16`.
- Fusion strategy: project both scales to the same channel width, upsample `stride=16`, concatenate, then refine with depthwise separable blocks.

## Training Decisions

- Required annotations: player bounding boxes for the central frame.
- Positive cell assignment: by bounding-box center.
- Objectness loss: focal loss.
- Box loss: `L1` or `SmoothL1`, optionally combined with IoU loss for positive cells only.
- Recommended initial total loss: `L_obj + 2.0 * L_box`.

## Postprocessing Decisions

- Decode all cells above an objectness threshold.
- Apply NMS.
- Filter detections by court polygon.
- Court inclusion rule: use the bottom-center point of each predicted box.
- Expand the court polygon slightly to avoid dropping players on boundary lines.
- Add simple sanity filters for minimum size, maximum size, and aspect ratio.

## Spectator Handling

- Do not hard-code exactly `12` outputs.
- Let the detector over-predict a little if needed.
- Remove spectators mainly with the court-zone filter and box-shape filters.
- If camera geometry is stable, keep one court polygon per camera or match.

## Scope Of This Step

- Added the model definition file only.
- Dataset, loss, decoding, training loop, and court-polygon filtering are the next implementation steps.
