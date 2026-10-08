"""Model survey PDF (run from the repo root: uv run --with reportlab python docs/report/build_models_pdf.py)."""
import os
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path.cwd() / "docs" / "landmark_models_survey.pdf"
ttf = Path(os.path.dirname(matplotlib.__file__)) / "mpl-data" / "fonts" / "ttf"
for name, file in [("DV", "DejaVuSans.ttf"), ("DVB", "DejaVuSans-Bold.ttf"), ("DVI", "DejaVuSans-Oblique.ttf")]:
    pdfmetrics.registerFont(TTFont(name, str(ttf / file)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")

ss = getSampleStyleSheet()
BLUE = colors.HexColor("#1f4e79")
body = ParagraphStyle("body", parent=ss["BodyText"], fontName="DV", fontSize=9.3, leading=13, spaceAfter=4)
small = ParagraphStyle("small", parent=body, fontSize=7.8, leading=10, textColor=colors.HexColor("#444444"))
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="DVB", fontSize=15, leading=19, textColor=BLUE, spaceAfter=6)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="DVB", fontSize=11.5, leading=14.5, textColor=BLUE,
                    spaceBefore=4, spaceAfter=3)
cell = ParagraphStyle("cell", parent=body, fontSize=8.4, leading=11, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="DVB")
W = A4[0] - 4 * cm


def P(t, s=body):
    return Paragraph(t, s)


def grid(rows, widths, header=True, style=cell, bg=None):
    t = Table([[P(c, style) if isinstance(c, str) else c for c in r] for r in rows], colWidths=widths,
              repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9a9a9a")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe5f1")))
    for (r, c) in (bg or []):
        st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor(c)))
    t.setStyle(TableStyle(st))
    return t


VERDICT_COLOR = {"USE NOW": "#cdebc9", "TRY NEXT": "#fbe7b3", "LATER STAGE": "#dfe7f2", "ALTERNATIVE": "#efe0f3",
                 "NOT FOR US": "#f4d4d4"}


def model_card(n, name, meta, rows, verdict, why):
    head = Table([[P(f"<b>{n}. {name}</b>", ParagraphStyle("mh", parent=body, fontName="DVB", fontSize=10.5,
                                                           textColor=colors.white)),
                   P(f"<b>{verdict}</b>", ParagraphStyle("v", parent=cell, alignment=TA_CENTER))]],
                 colWidths=[W - 3.4 * cm, 3.4 * cm])
    head.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), BLUE),
                              ("BACKGROUND", (1, 0), (1, 0), colors.HexColor(VERDICT_COLOR[verdict])),
                              ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("BOX", (0, 0), (-1, -1), 0.4, colors.grey)]))
    info = grid([[P("<b>" + k + "</b>", cell), P(v, cell)] for k, v in [("Facts", meta)] + rows +
                 [("Verdict for this project", why)]], [3.6 * cm, W - 3.6 * cm], header=False)
    return KeepTogether([head, info, Spacer(1, 9)])


def on_page(c, doc):
    c.saveState()
    c.setFont("DV", 7.5)
    c.setFillColor(colors.HexColor("#777777"))
    c.drawString(2 * cm, 1.2 * cm, "Landmark models for multi-camera fisheye 3D pose - survey (October 2026)")
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"page {doc.page}")
    c.restoreState()


s = []
s += [P("Landmark Extraction Models for Multi-Camera 3D Pose", ParagraphStyle("t", parent=h1, fontSize=20, leading=25)),
      P("Survey of the most recent models (2023 - October 2026) for extracting accurate body landmarks from our three "
        "fisheye security-camera videos, for offline processing and 3D use.", body),
      Spacer(1, 6),
      P("1. What we need - the selection criteria", h2),
      grid([["criterion", "why it matters here"],
            ["<b>Accuracy first, not speed</b>", "the videos are processed offline; hours of GPU time are acceptable"],
            ["<b>Precise 2D localisation in every view</b>", "3D comes from triangulating 2D points of 3 calibrated cameras; "
             "1 px of 2D error at 4 m distance is ~4 mm in 3D, and errors in different views do not cancel"],
            ["<b>Feet + body joints</b>", "the robot needs body and foot joints (heel, toes) now and joint angles later"],
            ["<b>Robust to our footage</b>", "ceiling-mounted fisheye cameras (top-down, distorted), person often seen from "
             "behind, a mop handle crossing the body, other people in the room"],
            ["<b>Consistent over time</b>", "the robot follows a trajectory - jitter and left/right swaps between frames "
             "are as harmful as a wrong frame"],
            ["<b>Usable in practice</b>", "public code + weights, runs on one GPU (8-24 GB), licence ok for research"]],
           [5.0 * cm, W - 5.0 * cm]),
      Spacer(1, 8),
      P("2. How these models find landmarks - the four families", h2),
      grid([["family", "how it works", "what it gives us", "examples"],
            ["<b>A. 2D keypoint detectors</b> (top-down)",
             "a person detector gives a box; the crop is resized; a network outputs, for every joint, a "
             "<i>heatmap</i> (probability of the joint at each pixel) or two 1D distributions along x and y (SimCC); "
             "the peak is the keypoint and its height the confidence",
             "precise 2D points per camera + confidences, which our triangulation turns into 3D",
             "Sapiens2, CIGPose, RTMW, ViTPose++, YOLO26-pose"],
            ["<b>B. Multi-view fusion / triangulation</b>",
             "combine the 2D evidence of several cameras into one 3D point: algebraic (weighted DLT, what we have), "
             "learned (fusing heatmaps or features across views), or with priors (bone lengths, temporal smoothness)",
             "3D joints in metres; rejection of a wrong view",
             "MEOM, algebraic-priors MV, Kineo, Pose2Sim"],
            ["<b>C. 3D body-model (mesh) recovery</b>",
             "a network regresses the parameters of a parametric body (SMPL-X or Meta's MHR): joint rotations + body shape "
             "+ camera; landmarks come from the fitted body",
             "a complete, anatomically valid skeleton with joint <b>angles</b>; plausible even when joints are hidden",
             "SAM 3D Body, SAM-Body4D, NLF, PromptHMR"],
            ["<b>D. Multi-view model fitting</b>",
             "optimise one body model so that its projections match the 2D keypoints of all cameras at once, plus "
             "smoothness and pose priors",
             "metric 3D + joint angles from all views together",
             "EasyMocap, Pose2Sim (OpenSim IK)"]],
           [3.2 * cm, 6.3 * cm, 3.9 * cm, W - 13.4 * cm]),
      Spacer(1, 6),
      P("<b>About video:</b> almost all of these models look at <b>one frame at a time</b>. Temporal consistency then "
        "comes from (a) multi-view triangulation (three cameras must agree), (b) filtering the 3D trajectories, or (c) "
        "the few video models (SAM-Body4D, PromptHMR-Vid, the algebraic-priors method) that look at several frames. "
        "<b>About fisheye:</b> none of the public models is trained on ceiling fisheye images; all are trained on "
        "ordinary photos. This is why our pipeline feeds them a <b>rectified virtual-camera crop</b> around the person - "
        "that turns every model below into a 'fisheye-capable' model. A top-view fisheye dataset (NToP570K) exists and "
        "fine-tuning ViTPose-B on it raised AP by 33 points - an option if the top-down viewpoint still hurts.", body),
      PageBreak()]

# --------------------------------------------------------------------------- A
s += [P("3. Family A - 2D keypoint detectors", h1)]
s.append(model_card(1, "Sapiens2 (Meta)",
    "ICLR 2026 · vision transformer foundation model pre-trained on 1 billion human images · sizes 0.4B / 0.8B / 1B / 5B · "
    "308 keypoints (body, feet, 40 hand, 243 face) · input 1024 x 768 · Sapiens2 licence (research ok)",
    [("How it finds landmarks", "top-down: the person crop goes through a very large ViT at high resolution; a heatmap "
      "head outputs 308 heatmaps (UDP decoding, sub-pixel peak); flip-test averages the image and its mirror"),
     ("Video", "per frame; no temporal model - consistency must come from triangulation and filtering"),
     ("Fisheye / our footage", "not trained on fisheye; on our rectified crops it gave the most confident and the "
      "steadiest keypoints (cam1 test: confidence 0.87, jitter 0.64 px)"),
     ("Accuracy", "1B: 80.4 mAP, 5B: 82.3 mAP on a 308-keypoint in-the-wild benchmark (RTMW-x: 70.2 on the same set)"),
     ("Practical", "public weights (no login); 0.8B needs ~2.8 GB in fp16, 0.83 frames/s on an RTX 2070; 1B on a 16 GB+ GPU")],
    "USE NOW", "<b>Our main model.</b> Best published accuracy for body + feet, already integrated and tested. On the "
    "GPU computer, use the 1B model if it has ≥16 GB."))
s.append(model_card(2, "CIGPose (CVPR 2026)",
    "March 2026 · Causal-Intervention Graph network on top of an RTMPose-style backbone · 133 COCO-WholeBody keypoints · "
    "Apache-2.0 · MMPose project, checkpoints on Google Drive",
    [("How it finds landmarks", "first predicts keypoints with an uncertainty; keypoints that look 'confounded' by the "
      "background or occlusion are replaced by learned canonical embeddings, then a hierarchical graph neural network "
      "refines all joints together so the skeleton stays anatomically plausible"),
     ("Video", "per frame"),
     ("Fisheye / our footage", "not trained on fisheye - use our rectified crops. Its strength (robustness to "
      "occlusion and clutter) fits our scenes: mop handle, desks, chairs"),
     ("Accuracy", "CIGPose-x 67.0 AP whole-body (67.5 with UBody) - new SOTA on COCO-WholeBody among RTMPose-type models"),
     ("Practical", "needs an MMPose environment (no ONNX yet); medium speed; 384 x 288 input")],
    "TRY NEXT", "Best candidate for a <b>second, independent opinion</b> next to Sapiens2 (better than RTMW, same keypoint "
    "format). Disagreements between Sapiens2 and CIGPose flag unreliable frames."))
s.append(model_card(3, "RTMW / RTMPose (OpenMMLab, via rtmlib)",
    "2023-2024 · CNN + SimCC head · 133 COCO-WholeBody keypoints · Apache-2.0 · ONNX, runs without PyTorch",
    [("How it finds landmarks", "predicts two 1D distributions per joint (one along x, one along y) instead of a 2D "
      "heatmap - fast and sub-pixel"),
     ("Video", "per frame (rtmlib has a simple tracker)"),
     ("Fisheye / our footage", "works on our crops; the 'cocktail13' RTMW-x release put the left toes on the head, the "
      "'cocktail14' release is correct; agrees with Sapiens2 within 1-4 px on all joints"),
     ("Accuracy", "70.1 AP whole-body (384 x 288)"),
     ("Practical", "~30 frames/s on an RTX 2070; standard 2D backend of Pose2Sim")],
    "USE NOW", "Already integrated as the <b>fast baseline and cross-check</b>. Not the most accurate."))
s.append(model_card(4, "ViTPose / ViTPose++ (Univ. Sydney)",
    "NeurIPS 2022 / TPAMI 2023 · plain vision transformer + simple heatmap decoder · sizes B, L, H, G (1B) · "
    "17 body keypoints (ViTPose++ also animals/whole-body heads) · Apache-2.0 · in Hugging Face transformers",
    [("How it finds landmarks", "top-down heatmaps from a plain ViT; multi-dataset training"),
     ("Video", "per frame"),
     ("Fisheye / our footage", "rectified crops needed; the top-view fisheye dataset NToP570K was built on ViTPose - a "
      "fine-tuned version is the most direct route to a 'ceiling-view' model"),
     ("Accuracy", "ViTPose-G 80.9 AP on COCO test-dev (body only); ViTPose++-H 78.5 AP"),
     ("Practical", "easy (transformers), but no foot keypoints in the standard models")],
    "ALTERNATIVE", "Strong body accuracy but <b>no feet</b>; Sapiens2 is the newer and more complete successor of the "
    "same idea. Relevant only if we fine-tune on top-view fisheye data."))
s.append(model_card(5, "YOLO26-pose (Ultralytics)",
    "2026 · single-stage detector that outputs boxes + keypoints in one pass, RLE keypoint loss · 17 body keypoints · AGPL-3.0",
    [("How it finds landmarks", "bottom-up/one-stage: every person and keypoint predicted from the full image at once"),
     ("Video", "per frame + built-in trackers (BoT-SORT/ByteTrack)"),
     ("Fisheye / our footage", "runs on raw fisheye frames (lower resolution per person, distorted)"),
     ("Accuracy", "YOLO26x-pose 71.6 AP (COCO body)"),
     ("Practical", "real-time, very easy")],
    "NOT FOR US", "Built for speed, not maximum accuracy, and <b>no foot keypoints</b>. We use the YOLO26x <b>detector</b> "
    "only to find and track people."))

# --------------------------------------------------------------------------- B
s += [PageBreak(), P("4. Family B - multi-view fusion and triangulation", h1)]
s.append(model_card(6, "MEOM - Multi-view Expected-OKS Maximisation",
    "arXiv, 31 Aug 2026 · newest work on multi-view triangulation · works on 2D heatmaps · code not announced yet",
    [("How it finds landmarks", "instead of taking one 2D point per view and intersecting rays (DLT), it places each 3D "
      "joint where the <b>probability mass of all views' heatmaps agrees</b> - so a second, hidden candidate in one "
      "view (multimodal heatmap under occlusion) is not lost"),
     ("Video", "per frame"),
     ("Accuracy", "19.1 mm MPJPE on Human3.6M with 3D supervision, better than volumetric methods at half the cost; "
      "strong on occluded data without 3D labels"),
     ("Practical", "needs heatmaps from the 2D model (Sapiens2 produces them) and calibrated cameras")],
    "LATER STAGE", "The natural <b>upgrade of our DLT triangulation</b> once the basic 3D pipeline runs; our backend "
    "can keep the heatmaps for this."))
s.append(model_card(7, "Unconstrained Multi-view Pose with Algebraic Priors",
    "arXiv, Apr 2026 · uncalibrated multi-view, transformer-based triangulation + Gröbner-basis geometry corrector + "
    "temporal equivariant rectifier · code not announced",
    [("How it finds landmarks", "fuses 2D keypoint tokens from all views with a transformer (no camera parameters), "
      "then enforces projective-geometry constraints and temporal coherence over the sequence"),
     ("Video", "<b>yes</b> - temporal module across frames"),
     ("Accuracy", "state of the art for uncalibrated multi-view; closes much of the gap to calibrated methods"),
     ("Practical", "research code status unclear")],
    "ALTERNATIVE", "Interesting because it needs <b>no extrinsic calibration</b>, but calibrated triangulation stays "
    "more accurate - we have (or will have) calibration."))
s.append(model_card(8, "Kineo - calibration-free metric motion capture",
    "arXiv Oct 2025 · open code (CC BY-NC-SA 4.0) · sparse, <b>unsynchronised and uncalibrated</b> consumer cameras",
    [("How it finds landmarks", "off-the-shelf 2D keypoints + dense point maps; a graph-based global optimisation "
      "jointly estimates camera poses, lens distortion (Brown-Conrady), time offsets and metric 3D keypoints"),
     ("Video", "<b>yes</b> - uses spatio-temporal keypoint sampling over the whole sequence"),
     ("Accuracy", "83-91 % lower world joint error than earlier calibration-free methods"),
     ("Practical", "processes 80 min of footage in ~36 min; designed for normal lenses (our fisheye would need our "
      "intrinsics or rectified views)")],
    "ALTERNATIVE", "A <b>backup for our two open steps</b> (camera positions and time sync): it solves both from the "
    "person's motion. Also useful to cross-check our own extrinsics."))
s.append(model_card(9, "Pose2Sim",
    "open-source pipeline (BSD-3), actively maintained, peer-reviewed accuracy studies · RTMPose 2D + weighted DLT + "
    "filtering + OpenSim inverse kinematics",
    [("How it finds landmarks", "2D per camera (RTMPose/RTMW), robust weighted triangulation with view rejection, "
      "Butterworth filtering, then a biomechanical skeleton is fitted (OpenSim) to get joint angles"),
     ("Video", "temporal filtering of 3D trajectories; synchronisation tool included"),
     ("Accuracy", "joint angles within ~2-6° of marker-based motion capture in validation studies"),
     ("Practical", "fisheye supported through its calibration; our framework already follows the same design")],
    "LATER STAGE", "The <b>reference design for our joint-angle stage</b>: we can export our 3D landmarks to its "
    "OpenSim kinematics step to get anatomically valid joint angles for the robot."))

# --------------------------------------------------------------------------- C/D
s += [PageBreak(), P("5. Family C/D - 3D body models and multi-view fitting", h1)]
s.append(model_card(10, "SAM 3D Body (Meta)",
    "CVPR 2026 · single-image full-body mesh recovery · Momentum Human Rig (MHR) body model, separate skeleton and "
    "shape · trained on 7 M annotated images · SAM licence",
    [("How it finds landmarks", "an encoder-decoder (DINOv3 / ViT-H backbone) predicts the MHR pose and shape; a "
      "separate hand decoder refines hands; it can be <b>prompted with 2D keypoints or a mask</b> to fix ambiguous "
      "cases or pick the right person"),
     ("Video", "per image (see SAM-Body4D for video); a multi-view extension triangulates its 2D keypoints"),
     ("Fisheye / our footage", "use rectified crops; prompting it with our triangulated/projected keypoints keeps all "
      "views consistent"),
     ("Accuracy", "state of the art single-view mesh recovery, strong generalisation in the wild"),
     ("Practical", "public code + checkpoints (access request on Hugging Face); large model")],
    "LATER STAGE", "Best candidate for <b>joint angles and a full skeleton</b>: fit MHR to our multi-view keypoints, "
    "or use its per-view estimate as a prior when joints are hidden."))
s.append(model_card(11, "SAM-Body4D",
    "arXiv Dec 2025 · training-free video extension of SAM 3D Body · open code",
    [("How it finds landmarks", "runs SAM 3D Body on identity-consistent segmentation masks across the video; an "
      "occlusion-aware module recovers hidden body parts from neighbouring frames"),
     ("Video", "<b>yes</b> - temporally stable mesh trajectories"),
     ("Accuracy", "more stable than per-frame SAM 3D Body on in-the-wild videos"),
     ("Practical", "heavy; single-view - depth/scale less reliable than our triangulation")],
    "ALTERNATIVE", "Good for <b>temporal smoothness and occlusion</b>; single-camera depth makes it a prior, not the "
    "final 3D measurement."))
s.append(model_card(12, "NLF - Neural Localizer Fields (MPI Informatics)",
    "NeurIPS 2024 · 3D pose and shape for <b>any point of the body</b> (any skeleton, any surface point) · PyTorch and "
    "TensorFlow models, research licence",
    [("How it finds landmarks", "a network outputs, for any queried body point, a 3D heatmap localiser - so one model "
      "gives SMPL joints, COCO joints, biomechanical marker positions, or mesh vertices"),
     ("Video", "per frame (batched)"),
     ("Fisheye / our footage", "accepts the camera intrinsics, so it works on our rectified crops; yields metric 3D in "
      "the camera frame"),
     ("Accuracy", "outperformed the state of the art on several 3D benchmarks at publication"),
     ("Practical", "public weights; flexible skeleton definition is ideal for a robot kinematic model")],
    "ALTERNATIVE", "Unique ability to output <b>exactly the joints the robot needs</b> (e.g. anatomical joint centres); "
    "useful as a 3D prior or to cross-check our triangulated skeleton."))
s.append(model_card(13, "PromptHMR (CVPR 2025) / PromptHMR-Vid",
    "promptable SMPL-X mesh recovery · video version with temporal transformer + SLAM for world coordinates",
    [("How it finds landmarks", "image + prompts (boxes, masks, text, interaction) → SMPL-X; the video version adds a "
      "temporal transformer for smooth motion"),
     ("Video", "<b>yes</b>"),
     ("Accuracy", "3DPW PA-MPJPE 35.5 mm (video version)"),
     ("Practical", "single-camera; SMPL-X licence")],
    "ALTERNATIVE", "Strong monocular video model, but our three calibrated cameras give real metric 3D; at most a prior."))
s.append(model_card(14, "EasyMocap (Zhejiang University)",
    "mature open-source toolbox · fits SMPL / SMPL+H / SMPL-X / MANO to 1-23 calibrated, synchronised cameras",
    [("How it finds landmarks", "2D keypoints from any detector, then one body model optimised against all views with "
      "pose priors and temporal smoothness"),
     ("Video", "<b>yes</b> - temporal smoothing terms"),
     ("Accuracy", "multi-view constraints make it metric and robust; quality depends on the 2D input"),
     ("Practical", "needs our calibration converted to its format; fisheye via rectified/undistorted keypoints")],
    "LATER STAGE", "Ready-made way to get a <b>full body model with joint angles</b> from our multi-view keypoints."))
s.append(model_card(15, "Top-view fisheye fine-tuning (NToP570K dataset)",
    "2024-2025 · 570 k synthetic top-view fisheye images rendered with NeRF, with 2D and 3D labels",
    [("How it finds landmarks", "not a new model: a dataset to fine-tune existing 2D/3D models (ViTPose, HybrIK) to the "
      "ceiling-camera viewpoint"),
     ("Accuracy", "fine-tuning ViTPose-B: +33.3 AP on top-view fisheye; HybrIK: -53.7 mm PA-MPJPE"),
     ("Practical", "requires training")],
    "ALTERNATIVE", "Only if our rectified crops still fail when the person is directly below a camera (strong top-down view)."))

# --------------------------------------------------------------------------- summary + recommendation
s += [PageBreak(), P("6. Comparison", h1),
      grid([["#", "model", "year", "type", "keypoints", "video", "3D use", "verdict"],
            ["1", "Sapiens2", "2026", "2D heatmap ViT", "308 (body, feet)", "frame", "triangulate", "<b>use now (main)</b>"],
            ["2", "CIGPose", "2026", "2D + graph net", "133", "frame", "triangulate", "<b>try next</b>"],
            ["3", "RTMW", "2024", "2D SimCC", "133", "frame", "triangulate", "use now (baseline)"],
            ["4", "ViTPose++", "2023", "2D heatmap ViT", "17", "frame", "triangulate", "alternative"],
            ["5", "YOLO26-pose", "2026", "1-stage 2D", "17", "frame+track", "triangulate", "detector only"],
            ["6", "MEOM", "2026", "heatmap fusion", "any", "frame", "3D directly", "later stage"],
            ["7", "Algebraic priors", "2026", "uncalibrated MV", "any", "yes", "3D directly", "alternative"],
            ["8", "Kineo", "2025", "calib-free MV", "any", "yes", "3D + calib + sync", "alternative / backup"],
            ["9", "Pose2Sim", "2022-26", "MV pipeline + IK", "RTMPose", "filter", "3D + angles", "later stage"],
            ["10", "SAM 3D Body", "2026", "mesh (MHR)", "full body", "frame", "prior / fit", "later stage"],
            ["11", "SAM-Body4D", "2025", "video mesh", "full body", "yes", "prior", "alternative"],
            ["12", "NLF", "2024", "3D any point", "any", "frame", "camera-space 3D", "alternative"],
            ["13", "PromptHMR-Vid", "2025", "video SMPL-X", "SMPL-X", "yes", "prior", "alternative"],
            ["14", "EasyMocap", "2021-25", "MV body fitting", "SMPL-X", "yes", "3D + angles", "later stage"],
            ["15", "NToP570K", "2024-25", "fine-tune data", "-", "-", "-", "if needed"]],
           [0.9 * cm, 2.7 * cm, 1.4 * cm, 2.9 * cm, 2.3 * cm, 1.9 * cm, 2.6 * cm, W - 14.7 * cm],
           bg=[(1, "#e6f4e3"), (2, "#fdf3d8"), (3, "#e6f4e3")]),
      Spacer(1, 6),
      P("Accuracy numbers come from different benchmarks (COCO body, COCO-WholeBody, Sapiens' 308-keypoint set, "
        "Human3.6M, 3DPW) and are <b>not directly comparable</b> across rows; the decisive test for our project is the "
        "multi-view reprojection error and bone-length stability on our own footage.", small),
      Spacer(1, 6),
      P("7. Recommended accuracy-first pipeline", h1),
      grid([["stage", "choice", "reason"],
            ["2D landmarks", "<b>Sapiens2</b> (1B on a ≥16 GB GPU, else 0.8B) on rectified virtual-camera crops",
             "highest published accuracy with feet; already integrated and tested"],
            ["second opinion", "<b>CIGPose-x</b> (replacing RTMW as cross-check)", "independent, occlusion-robust, "
             "more accurate than RTMW; disagreement marks bad frames"],
            ["3D", "weighted DLT with view rejection (implemented) → later <b>MEOM-style heatmap fusion</b>",
             "uses all three calibrated views; heatmap fusion keeps occluded alternatives"],
            ["calibration + sync", "reference object + our tools; <b>Kineo</b> as cross-check / backup",
             "Kineo estimates camera poses and time offsets from the person's motion"],
            ["time consistency", "3D filtering; optional SAM-Body4D prior", "robot needs smooth trajectories"],
            ["joint angles", "<b>SAM 3D Body / MHR</b> or <b>Pose2Sim-OpenSim</b> skeleton fit to the 3D landmarks",
             "anatomically valid angles with constant bone lengths"]],
           [3.0 * cm, 6.8 * cm, W - 9.8 * cm]),
      Spacer(1, 8),
      P("Sources", h2),
      P("Sapiens2: arxiv.org/abs/2604.21681, github.com/facebookresearch/sapiens2 · CIGPose: arxiv.org/abs/2603.09418, "
        "github.com/53mins/CIGPose · RTMW: arxiv.org/abs/2407.08634, github.com/Tau-J/rtmlib · ViTPose/ViTPose++: "
        "arxiv.org/abs/2204.12484, arxiv.org/abs/2212.04246 · YOLO26: docs.ultralytics.com/models/yolo26 · MEOM: "
        "arxiv.org/abs/2608.30521 · Algebraic priors: arxiv.org/abs/2604.24312 · Kineo: arxiv.org/abs/2510.24464 · "
        "Pose2Sim: github.com/perfanalytics/pose2sim · SAM 3D Body: arxiv.org/abs/2602.15989, "
        "github.com/facebookresearch/sam-3d-body · SAM-Body4D: arxiv.org/abs/2512.08406 · NLF: arxiv.org/abs/2407.07532, "
        "github.com/isarandi/nlf · PromptHMR: arxiv.org/abs/2504.06397 · EasyMocap: github.com/zju3dv/EasyMocap · "
        "NToP570K / top-view fisheye: arxiv.org/abs/2402.18196", small)]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                        bottomMargin=1.8 * cm, title="Landmark extraction models - survey", author="moppose project")
doc.build(s, onFirstPage=on_page, onLaterPages=on_page)
print("wrote", OUT, f"{OUT.stat().st_size / 1e3:.0f} kB")
