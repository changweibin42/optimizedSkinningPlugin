# MOS / OCD Curve Sampling

In Maya 2024, select **Algorithm Type → MOS** and **MOS Curve Mode → MOS / OCD Curve Sampling**. The new mode is enum value 4 and is the default for newly created nodes. A scene with an explicitly saved mode 3 continues to use LBS Radial Compensation until you switch modes. Start a shape comparison with **MOS Weight** set to 1 and all secondary controls at 0. Set MOS Weight to 0 for the affine LBS reference.

## Deformation

- At bind, triangulation is irrelevant to the curve attachment. A vertex selects the closest cubic B-spline guide among guides containing a weighted influence. The cached attachment contains its parameter, four active basis coefficients and a single local offset. The offset is solved using the weighted bind frames and control centers in US9786083B2, Eq. 9.
- At evaluation, the posed centers and frames generate each vertex's virtual offset control points (US8400455B2 and US9786083B2, Eq. 5). The bind coefficients interpolate those points. Vertices with only non-guide influences or a singular bind frame use their affine skinning result. Guide coverage and MOS Weight control the blend with that result.
- The guide carries transported frames. MOS Twist Weight controls their axial twist, and the existing MOS Scale Mode / Weight control scaling in the local guide frame. A closed guide samples periodically with duplicated initial controls and a roll correction calculated from the transported closing frame.
- Each active control can use multiple bend transforms. Their independent slide and bulge weights determine their individual contributions, as described in US9418465B2. The automatic in-vector bisects the fold. A manual in-vector is in the selected driver's bind joint coordinates and follows the driver in the current pose. Secondary strength responds to the change in bend from the bind pose, with 180 degrees of change giving unit response on a straight bind.
- Positive inner Slide moves its limb toward the bend along the contributing control's twist axis. Positive outer Slide moves its limb away. Slide Ortho extends the chosen sector radially, Center Slide edits the center of the fold, and Side Bulge extends the positive and negative lateral sectors. These additional channel names, signs and envelopes are specific to this plug-in. They can be checked one at a time at 45, 90 and 140 degrees.
- The old two-bone shape guide, collision behavior and center-sideways channels are independent of this solver. The public patent illustrations do not specify the curves, weights, UI values or poses used for a particular demonstration.

## Optional guide authoring

The default uses open guides inferred from the skinned joint hierarchy. An open guide has a terminal segment owned by its last influence. Its length comes from mesh points past the last pivot with nontrivial weight on that influence, or from the preceding segment when no such point exists. Existing node-wide Slide and Bulge attributes drive the default controls.

These typed Maya attributes are available for explicit setups. Guides, joint slots within a guide, and controls within a guide are zero-based. Use logical skin matrix indices wherever a record calls for an influence ID.

| Attribute | Type | Records |
| --- | --- | --- |
| `mosGuideInfluences` | `Int32Array` | Ordered list of influence IDs. Separate guides with `-1`. Repeat the first ID at the end of a guide to close its loop. |
| `mosGuideControls` | `doubleArray` | `[guide, jointSlot, worldBindX, worldBindY, worldBindZ]`. Records are in curve order. A guide with any authored controls requires at least four. |
| `mosControlDrivers` | `doubleArray` | `[guide, control, driverJointSlot, slideWeight, bulgeWeight]`. Multiple records replace that control's generated drivers. Slide and bulge weights are normalized independently. A row with two zero weights disables that control's secondary response. |
| `mosControlValues` | `doubleArray` | `[guide, control, side, bulgeIn, bulgeOut, bulgeSide, slideIn, slideOut, orthoIn, orthoOut, centerIn, centerOut]`. Side 0 is previous, side 1 is next. Each row overrides that side's node-wide values for the specified control. |
| `mosDriverInVectors` | `doubleArray` | `[influenceId, jointSpaceX, jointSpaceY, jointSpaceZ]`. A zero vector requests the automatic in-vector for that influence. |

For example, in Maya Python, for a skin node whose upper-arm and forearm influences use matrix indices 2 and 9:

```python
import maya.cmds as cmds

node = "optimizedSkinCluster1"
cmds.setAttr(node + ".algorithmType", 3)
cmds.setAttr(node + ".mosCurveMode", 4)
cmds.setAttr(node + ".mosGuideInfluences", 2, 2, 9, type="Int32Array")
cmds.setAttr(node + ".mosInfluenceRadius", 1.0)
```

## Checks

`tests/test_mos_curve_sampling.cpp` checks rest reconstruction, posed samples against explicit deformed offset curves, single-channel directions, response through 140 degrees, weighted multiple drivers and the closed seam. Compile it with `src/mos_curve_sampling.cpp`, `src/ocd_math.cpp` and `-I src` using C++17.

With the compiled plug-in loaded in Maya, run the scene-level smoke test from the checkout:

```python
import runpy
runpy.run_path(r"D:\optimizedSkinningPlugin\mos-ocd-curve-sampling\tests\maya_mos_curve_smoke.py")["smoke_test"]()
```

Public descriptions: [Sony OCD](https://patents.google.com/patent/US8400455B2/en), [DreamWorks MOS](https://patents.google.com/patent/US9786083B2/en), [MOS secondary operations](https://patents.google.com/patent/US9418465B2/en).
