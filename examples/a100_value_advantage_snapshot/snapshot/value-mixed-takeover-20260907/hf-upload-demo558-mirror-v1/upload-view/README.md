# PiperX advantage-selected teleoperation segments

Only pure human demonstrations. Value checkpoint step1500 (mixed demo+HIL); no HIL frames in this export.

A50 ranked globally across 558 source episodes, top10% AND A>0. Every selected start expands to [t,t+20); overlaps and adjacency merge. Each disconnected component is a separate output episode. Interior frames need not themselves be top10%.

Output: 7146 segments, 322261 frames, 2.983898 hours at30FPS.

Three camera streams and action/state are physically sliced; files pack segments from one source episode, with metadata separating episode boundaries. Videos re-encoded H264 CRF18/fast 640x480; PTS and full decode validated. Action/state retained exactly. Stats recalculated using LeRobot v0.6.1, absolute actions, includes q01/q10/q50/q90/q99.

Traceability: selection/segments.json, selection/frames/*.parquet, selection/protocol.json. Original Value train/test membership retained per segment. This all558 dataset is not an untouched Value test set. Clip endings are cut boundaries, NOT task-success labels. No automatic policy training or upload was performed.

Using every frame as a policy training start is not the same as restricting starts to selected anchors; use selection annotations if that is required. Policy action horizon remains a separate training setting; this dataset does not change it.
