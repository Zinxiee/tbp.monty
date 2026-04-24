Log confirms clean settling (0-2.7 mm errors, zero invalid pixels, no oscillation, burst ran 5 frames cleanly — noting actual burst_read_ms=400-800ms, so A010 runs closer to 10 Hz than 25 Hz, not a defect but worth knowing). Settling/motion blur ruled out as dominant cause.

Key findings from exploration:

Finding	~ Impact
12 cm is below reliable range — empirically, A010 noise doubles below 15 cm vs at 15 cm	~ Primary hypothesis now
Settling is clean (2-3 mm gap, 0 ms timeouts, 0 invalid pixels)	~ Settling is not the issue
30 nodes from 36 obs — Fix B merging is working (no node explosion)	~ Tolerance relaxation worked
Curvature features show K1 up to +200, K2 down to -206 m⁻¹ — still chaos ~ Curvature is unrecoverable on this hardware
Burst runs at ~10 Hz actual, not 25 Hz ~ Each burst takes ~500 ms; settle is fine