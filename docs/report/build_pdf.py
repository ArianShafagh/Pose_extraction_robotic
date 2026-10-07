"""Build the progress-report PDF (run from the repo root with `uv run --with reportlab python ...`)."""
import json
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
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

HERE = Path(__file__).parent
FIG = HERE / "fig"
OUT = Path.cwd() / "docs" / "moppose_progress_report.pdf"
stats = json.loads((FIG / "stats.json").read_text())

# fonts with Greek letters / superscripts
ttf = Path(os.path.dirname(matplotlib.__file__)) / "mpl-data" / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(ttf / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", str(ttf / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DVI", str(ttf / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DVM", str(ttf / "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")

ss = getSampleStyleSheet()
BLUE = colors.HexColor("#1f4e79")
body = ParagraphStyle("body", parent=ss["BodyText"], fontName="DV", fontSize=9.6, leading=13.6, spaceAfter=5)
small = ParagraphStyle("small", parent=body, fontSize=8.2, leading=10.5, textColor=colors.HexColor("#444444"))
cap = ParagraphStyle("cap", parent=small, alignment=TA_CENTER, spaceAfter=10)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="DVB", fontSize=15.5, leading=19, textColor=BLUE,
                    spaceBefore=6, spaceAfter=7)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="DVB", fontSize=11.6, leading=15, textColor=BLUE,
                    spaceBefore=8, spaceAfter=4)
mono = ParagraphStyle("mono", parent=body, fontName="DVM", fontSize=8.4, leading=11, backColor=colors.HexColor("#f3f3f3"),
                      borderPadding=4, spaceBefore=3, spaceAfter=7)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=2)
box_style = ParagraphStyle("box", parent=body, fontSize=9.2, leading=12.8)

W = A4[0] - 4 * cm


def P(t, s=body):
    return Paragraph(t, s)


def B(items):
    return [Paragraph(i, bullet, bulletText="•") for i in items]


def fig(name, width_cm, caption):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(FIG / name)).getSize()
    w = width_cm * cm
    return KeepTogether([Image(str(FIG / name), width=w, height=w * ih / iw), P(caption, cap)])


def table(rows, widths, header=True, font=8.6):
    tc = ParagraphStyle("tc", parent=body, fontSize=font, leading=font + 2.6, spaceAfter=0)
    t = Table([[P(c, tc) if isinstance(c, str) else c for c in r] for r in rows],
              colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe5f1"))]
    t.setStyle(TableStyle(st))
    return t


def img(path, width):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(path)).getSize()
    return Image(str(path), width=width, height=width * ih / iw)


def note(title, text, color="#eef5ea"):
    t = Table([[P(f"<b>{title}</b><br/>{text}", box_style)]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(color)),
                           ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#aaaaaa")),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return KeepTogether([t, Spacer(1, 8)])


def on_page(c, doc):
    c.saveState()
    c.setFont("DV", 7.5)
    c.setFillColor(colors.HexColor("#777777"))
    c.drawString(2 * cm, 1.2 * cm, "Multi-camera fisheye 3D pose of a mopping person - progress report")
    c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"page {doc.page}")
    c.restoreState()


s = []
# ---------------------------------------------------------------------------- title
s += [Spacer(1, 3.2 * cm),
      P("Multi-Camera 3D Pose Extraction<br/>of a Person Mopping", ParagraphStyle("t", parent=h1, fontSize=24, leading=30,
                                                                                alignment=TA_CENTER)),
      Spacer(1, 0.4 * cm),
      P("using three fisheye security cameras - towards a robot that shadows human cleaning motion",
        ParagraphStyle("st", parent=body, fontSize=12, leading=16, alignment=TA_CENTER, textColor=colors.HexColor("#444"))),
      Spacer(1, 1.2 * cm),
      P("Progress report - October 2026", ParagraphStyle("d", parent=body, fontSize=11, alignment=TA_CENTER)),
      Spacer(1, 1.5 * cm),
      fig("pipeline.png", 16.5, "Figure 1 - Overall pipeline. Green: implemented and tested on the real recordings. "
                                 "Yellow: next steps (triangulation code already implemented and tested on synthetic data)."),
      Spacer(1, 0.6 * cm),
      note("Summary",
           "Three ceiling-mounted fisheye security cameras film a person mopping an office. The goal is a clean 3D "
           "trajectory of the body joints (and later joint angles) that a robot can imitate. So far: "
           "(1) each camera's fisheye lens is calibrated with a ChArUco board (reprojection RMS 0.76-0.95 px, "
           "covering 96-99 % of the area where people move); (2) a 2D pose pipeline finds the mopper among the other "
           "people, removes the fisheye distortion around them with a virtual pinhole camera and estimates 23 body + "
           "foot keypoints with two state-of-the-art models (Sapiens2 and RTMW) that agree within 1-4 px. "
           "Next: camera positions in a common world frame, time synchronisation and triangulation to 3D."),
      PageBreak()]

# ---------------------------------------------------------------------------- 1 goal
s += [P("1. Goal and overall approach", h1),
      P("The long-term aim is a robot that <b>shadows the movements of a person mopping</b>. The robot needs, for every "
        "moment in time, the 3D positions of the person's joints (shoulders, elbows, wrists, hips, knees, ankles, feet) "
        "in metres, and later the joint angles. This project builds the <b>measurement framework</b> that produces "
        "these landmarks from video; the robot control itself is a separate, later part."),
      P("<b>Why several cameras?</b> A single camera only sees a 2D projection: depth is ambiguous and body parts get "
        "hidden. If the same joint is seen from two or more calibrated cameras, its 3D position is the intersection of "
        "the viewing rays (<i>triangulation</i>). Three cameras also give redundancy when one view is occluded."),
      P("<b>The pipeline</b> (Figure 1) has these stages:"),
      *B(["<b>Intrinsic calibration</b> - per camera: focal length, image centre and fisheye distortion, so each pixel "
          "can be turned into a viewing ray. <i>(done)</i>",
          "<b>Person tracking</b> - find the mopper among the other people in the room. <i>(done)</i>",
          "<b>2D pose estimation</b> - body keypoints in every frame of every camera, robust to the fisheye "
          "distortion. <i>(done, full-length runs on a GPU computer)</i>",
          "<b>Extrinsic calibration</b> - position and orientation of each camera in one common world frame, from an "
          "object seen by all three cameras. <i>(next)</i>",
          "<b>Time synchronisation</b> - the cameras record independently; their videos must be aligned in time. "
          "<i>(next)</i>",
          "<b>Triangulation, filtering, joint angles</b> - 3D landmarks over time, smoothed, then a skeleton fit. "
          "<i>(triangulation implemented and tested; the rest next)</i>"]),
      Spacer(1, 6),
      P("Everything is implemented as one Python package (<font name='DVM'>moppose</font>) with a command-line tool, "
        "automated tests on synthetic data, and all results under version control (git)."),
      ]

# ---------------------------------------------------------------------------- 2 data
s += [P("2. Hardware and recordings", h1),
      table([["", "value"],
             ["Cameras", "3 ceiling-mounted fisheye security cameras: cam1, cam3, cam4"],
             ["Mopping videos", "1280 x 1024 px, H.264 (.mkv), about 593 s each, recorded separately by each camera"],
             ["Frame rate", "file header says 15 fps, but the <b>real rate is ~30 fps and variable</b> (frame intervals 16-51 ms); "
                            "dropped frames: cam1 3, cam3 4, cam4 23"],
             ["Calibration board", "ChArUco, A3 print, 3 x 2 squares (see section 3.2)"],
             ["Board videos", "one per camera, 158-210 s; cam3's was exported at 1920 x 1080 (see 3.4)"],
             ["Computer", "laptop with NVIDIA RTX 2070 (8 GB); heavy models run on a stronger GPU PC"]],
            [3.2 * cm, W - 3.2 * cm]),
      Spacer(1, 6),
      note("Finding: the frame rate in the file cannot be trusted",
           "Security cameras record with a variable frame rate and drop frames. Using frame_number / fps as the time "
           "would drift by seconds over 10 minutes and break the synchronisation of the three cameras. Therefore the "
           "framework reads the <b>real timestamp of every frame</b> from the video container and uses it everywhere.",
           "#fff6e0"),
      fig("mopping_sheet.jpg", 15.5, "Figure 2 - Frames of the cam1 mopping recording (time stamps in seconds). "
                                       "Strong fisheye distortion is visible at the image borders; several other "
                                       "people sit in the room."),
      PageBreak()]

# ---------------------------------------------------------------------------- 3 calibration
s += [P("3. Step 1 - Lens (intrinsic) calibration", h1),
      P("3.1 What is calibrated and why", h2),
      P("A camera maps a 3D direction to a pixel. For a normal (pinhole) lens a ray at angle θ from the optical axis "
        "lands at distance r = f·tan θ from the image centre; f is the focal length in pixels. A fisheye lens "
        "compresses wide angles to fit a large field of view, so the image is strongly curved. We use the "
        "<b>equidistant fisheye model</b> of OpenCV:"),
      P("r(θ) = f · θ · (1 + k<sub>1</sub>θ<super>2</super> + k<sub>2</sub>θ<super>4</super> + "
        "k<sub>3</sub>θ<super>6</super> + k<sub>4</sub>θ<super>8</super>)", ParagraphStyle("eq", parent=body,
                                                                                         alignment=TA_CENTER, fontSize=10.5)),
      P("The <b>intrinsic parameters</b> are the focal lengths f<sub>x</sub>, f<sub>y</sub>, the principal point "
        "(c<sub>x</sub>, c<sub>y</sub>) - where the optical axis hits the image - and the distortion coefficients "
        "k<sub>1</sub>..k<sub>4</sub>. Once known, every pixel can be converted into a 3D viewing ray, which is what "
        "triangulation needs. As a comparison the framework also fits the standard pinhole model with rational "
        "distortion (8 coefficients) and keeps the fisheye model unless the pinhole fit is clearly (>10 %) better."),
      P("Calibration works by filming a flat pattern with exactly known geometry (the board) in many positions. For "
        "each frame the detected board points in the image and their known positions on the board give equations; "
        "a non-linear least-squares optimisation finds the lens parameters (and the board pose in each frame) that "
        "minimise the <b>reprojection error</b>: the pixel distance between detected points and the points predicted "
        "by the model. Its root mean square (RMS) is the main quality number."),
      P("3.2 The ChArUco board", h2),
      table([[img(FIG / "board.jpg", 6.2 * cm),
              P("<b>3 x 2 squares</b>, square 115 mm, marker 86.2 mm, ArUco dictionary DICT_4X4_50 (marker ids 0-2), "
                "printed on A3. A ChArUco board combines a chessboard (precise corners) with ArUco markers (each corner "
                "can be identified even if the board is only partly visible).<br/><br/>"
                "The parameters were <b>verified against the PDF</b>: OpenCV's own board generator with these values "
                "reproduces the PDF pixel for pixel.", body)]],
            [6.6 * cm, W - 6.6 * cm], header=False),
      Spacer(1, 6),
      note("Problem 1 - a 3 x 2 board has only 2 chessboard corners",
           "Calibration needs many well-spread points per frame. Solution: the 4 corners of each of the 3 markers are "
           "used as additional points (their 3D board positions are known too), giving <b>up to 14 points per frame</b>. "
           "Synthetic tests showed that with so few points per frame the quality comes from the number and variety of "
           "frames: ~40 frames gave ~0.45° ray error, ~100 frames ~0.15°.", "#fff6e0"),
      note("Problem 2 - a half-pixel error inside OpenCV 4.11",
           "On synthetic images with known ground truth, OpenCV's ChArUco detector returned the chessboard corners "
           "shifted by <b>+0.5 px</b> in x and y, while the marker corners were unbiased. Mixing both point types made "
           "the focal length 1.4 % wrong. The framework now measures this offset at start-up on a rendered board with "
           "known corners and subtracts it, so the correction adapts to whatever OpenCV version is installed. "
           "Afterwards the focal length was recovered within 1 %.", "#fff6e0"),
      P("3.3 Calibration procedure (command <font name='DVM'>calib-intrinsics</font>)", h2),
      *B(["sample one frame every 0.25 s of the board video; detect the board",
          "reject frames with fewer than 8 points, motion blur (variance of the Laplacian below a threshold) or "
          "points nearly on a line (no geometric information)",
          "choose ~80 frames that cover the image evenly (greedy selection over an 8 x 8 grid with diminishing "
          "returns, and no two frames within 0.5 s)",
          "fit the fisheye and the pinhole model; iteratively drop frames whose own error is far above the median",
          "write a report: coverage map, per-frame errors, undistorted image, and a usability verdict"]),
      Spacer(1, 4),
      P("3.4 Different video formats", h2),
      P("Some board videos were exported at 1920 x 1080 with black side bars, whereas the mopping videos are the "
        "camera's native 1280 x 1024. A calibration is only valid for one pixel grid, so the framework detects the "
        "active picture area, crops the bars and scales the board frames to 1280 x 1024 before detection. The mapping "
        "is verified automatically by matching the static room background (SIFT features) between the board video and "
        "the mopping video: the residual shift was <b>0.4-0.9 px</b>."),
      P("3.5 First attempt - and why it failed", h2),
      P("In the first board recordings the board was held in the middle of the image only (it reached 39-46 % of the "
        "distance to the image corners). The RMS error looked excellent (0.48-0.72 px) - but only because the model "
        "had to fit the centre. Figure 3 shows the consequence for cam1: the fitted lens curve (orange, dashed) "
        "follows the true one up to ~30° and then <b>folds back</b>, which is physically impossible. Undistorting the "
        "image with it produced a black image. <b>A low RMS does not prove a good calibration; coverage matters.</b>"),
      fig("lens_curve.png", 13.5, "Figure 3 - Lens curves of cam1. Orange: first attempt (board only in the centre) - "
                                  "the polynomial folds back beyond the measured region. Blue: final calibration, "
                                  "monotonic over the whole image (measured up to 551 px, extrapolated beyond)."),
      P("An automatic check was added: it reports how far from the centre the board was seen and whether the "
        "distortion curve stays monotonic, and prints USABLE / NOT USABLE. The board was then re-recorded closer to "
        "the cameras and moved over the image."),
      KeepTogether([P("3.6 Final calibration results", h2),
      table([["camera", "board frames found", "RMS", "f<sub>x</sub> / f<sub>y</sub> [px]",
              "c<sub>x</sub>, c<sub>y</sub> [px]", "board reach", "work area covered"],
             ["cam1", "719", "0.84 px", "911 / 917", "596, 490", "63 %", "<b>99 %</b>"],
             ["cam3", "595", "0.95 px", "932 / 939", "620, 541", "66 %", "<b>98 %</b>"],
             ["cam4", "783", "0.76 px", "902 / 907", "636, 505", "64 %", "<b>96 %</b>"]],
            [1.6 * cm, 2.4 * cm, 1.6 * cm, 2.5 * cm, 2.4 * cm, 2.0 * cm, W - 12.5 * cm])]),
      Spacer(1, 6),
      P("The image corners still were not reached - but they show ceiling and walls. What matters is the area where "
        "the person moves. The framework measures it from the mopping video with background subtraction and checks "
        "how much of it lies inside the board-covered area: <b>96-99 %</b>. The fisheye and pinhole fits also agree "
        "now (principal point within 5-10 px, focal length within 1 %), and straight edges in the room become straight "
        "after undistortion (Figure 5)."),
      fig("cam1_work_area.jpg", 10.5, "Figure 4 - cam1: heat map of where people move in the mopping video; white "
                                      "outline = area covered by board points. 99 % of the motion is inside."),
      fig("cam4_undistort.jpg", 13.5, "Figure 5 - cam4: original fisheye frame and undistorted versions "
                                      "(balance 0 / 0.5 / 1 = how much of the wide field of view is kept). "
                                      "Walls, cabinet and window frames become straight."),
      P("The calibrations, detected board points and report images are stored in the repository "
        "(<font name='DVM'>calibration/session1/</font>, git tag <font name='DVM'>calib-session1</font>); "
        "parameter values are listed in the appendix.", small),
      PageBreak()]

# ---------------------------------------------------------------------------- 4 pose
s += [P("4. Step 2 - 2D pose estimation", h1),
      P("4.1 Choice of models", h2),
      P("The videos are processed offline, so <b>accuracy has priority over speed</b>. The state of the art was "
        "reviewed in October 2026:"),
      table([["model", "keypoints", "reported accuracy", "role here"],
             ["<b>Sapiens2-0.8B</b> (Meta, ICLR 2026)", "308: body, feet, hands, face",
              "Sapiens2-1B 80.4 mAP, 5B 82.3 mAP on a 308-keypoint in-the-wild benchmark (0.8B: not reported separately)",
              "<b>main model</b>; vision transformer at 1024 x 768 input"],
             ["<b>RTMW</b> (rtmlib, ONNX)", "133: COCO-WholeBody", "70.1 AP (COCO-WholeBody)",
              "fast baseline for cross-checking; also used by Pose2Sim"],
             ["YOLO26x (Ultralytics)", "person boxes", "-", "person detection + tracking"],
             ["YOLO26x-pose", "17 body", "71.6 AP (COCO body)", "not used: no foot points"],
             ["SAM 3D Body (Meta, CVPR 2026)", "3D mesh per image", "single-view 3D",
              "candidate for the later joint-angle stage"]],
            [3.6 * cm, 2.9 * cm, 5.2 * cm, W - 11.7 * cm]),
      Spacer(1, 6),
      P("For the robot we keep a canonical set of <b>23 joints</b>: the 17 COCO body points (nose, eyes, ears, "
        "shoulders, elbows, wrists, hips, knees, ankles) plus big toe, small toe and heel on each foot. All other "
        "keypoints of the models (e.g. hands) are stored as well, for later use."),
      P("4.2 Finding the person who mops", h2),
      P("YOLO26x detects every person in each raw frame and the BoT-SORT tracker links the detections over time into "
        "tracks. The office contains several seated people, so the mopper is selected as the track whose <b>feet travel "
        "the farthest</b> (bottom-centre of the box, median-smoothed). Trackers sometimes split one person into several "
        "tracks (occlusion); pieces are joined when one ends close in time (&lt; 3 s) and space (&lt; 200 px) to where the "
        "next begins. A preview image lets the user confirm the choice or force a track id."),
      fig("tracks.jpg", 11, "Figure 6 - cam1, first 20 s: six tracks; the mopper (green, track 2) is chosen "
                            "automatically, the five seated people (grey) are ignored."),
      P("4.3 Removing the fisheye distortion around the person - the virtual camera", h2),
      P("Pose networks are trained on ordinary photographs. Near the edge of a fisheye image a person is bent and "
        "squeezed, which degrades the keypoints. Undistorting the whole frame is not possible for a wide fisheye "
        "(the borders would be stretched enormously). Instead, for every frame a <b>virtual pinhole camera</b> is "
        "placed at the same optical centre and rotated to look straight at the person:"),
      *B(["the person's bounding-box outline is converted to viewing rays with the calibration",
          "the virtual camera's axis points at the angular centre of these rays, with the image 'up' kept upright",
          "its focal length is chosen so the person fills the crop with a margin (the models add 25 % themselves)",
          "for every crop pixel the corresponding ray is projected through the real fisheye model, and the raw image "
          "is resampled there (a remapping table)",
          "keypoints found in the crop are converted back: crop pixel → ray → raw fisheye pixel. "
          "This round trip is exact to &lt; 0.05 px (unit test)."]),
      fig("virtual_crop.jpg", 15.5, "Figure 7 - Left: raw fisheye frame; green = detected person box, yellow = region "
                                    "seen by the virtual camera. Right: the rectified, upright 768 x 1024 crop given to "
                                    "the network, with the Sapiens2 skeleton."),
      PageBreak(),
      P("4.4 Results on the real footage (cam1, first 20 s, 301 frames)", h2),
      table([["", "Sapiens2-0.8B", "RTMW"],
             ["mean keypoint confidence", f"<b>{stats['sap_conf']:.2f}</b>", f"{stats['rt_conf']:.2f}"],
             ["frame-to-frame jitter (median 2nd difference)", f"<b>{stats['sap_jit']:.2f} px</b>", f"{stats['rt_jit']:.2f} px"],
             ["speed on RTX 2070 (8 GB), incl. flip test", "0.83 frames/s", "~30 frames/s"],
             ["GPU memory", "2.8 GB (fp16)", "small"]],
            [7.5 * cm, 4.2 * cm, W - 11.7 * cm]),
      Spacer(1, 6),
      P("Without ground truth, the two independent models serve as a cross-check: they agree within <b>1-4 px on all 23 "
        "joints</b> (Figure 9). Sapiens2 is more confident and steadier, so it is the main model; RTMW remains as a fast "
        "fallback and sanity check. The decisive accuracy test comes after triangulation: the reprojection error of "
        "the 3D joints in all three cameras and the constancy of bone lengths over time."),
      fig("pose_preview.jpg", 15, "Figure 8 - cam1: twelve frames over 20 s, zoomed on the mopper. Pink = Sapiens2, "
                                  "orange = RTMW (mostly hidden under pink because they agree)."),
      fig("agreement.png", 13, "Figure 9 - Median distance between Sapiens2 and RTMW per joint (raw image pixels)."),
      note("Problems found during testing and how they were solved",
           "• The RTMW-x 'cocktail13' release placed the <b>left toes on the head</b> on this footage (hundreds of pixels "
           "from the ankle, with high confidence). Switched to the RTMW 'cocktail14' release; its feet agree with "
           "Sapiens2.<br/>"
           "• RTMW returns unbounded confidence scores (~3-8); they are mapped to 0-1 with s/(1+s), raw values kept.<br/>"
           "• A CPU-only ONNX Runtime package silently replaced the GPU version (same module name); it is now excluded "
           "and the GPU build is pinned to CUDA 12 to match PyTorch.", "#fff6e0"),
      P("4.5 Computation", h2),
      P("Each camera has ~17 800 frames. On the laptop Sapiens2 would need about 6 hours per camera, so the full run is "
        "done on a stronger Windows GPU computer. The code is moved as one git bundle; two commands install everything "
        "and download the models; results are written in resumable chunks, so a run can be stopped and continued; only "
        "small result files (a few MB) come back. Each result file contains, per frame, the real timestamp, the 23 "
        "joints in raw fisheye pixels, their confidences, and all native keypoints of the model."),
      PageBreak()]

# ---------------------------------------------------------------------------- 5 triangulation
s += [P("5. Step 3 - Triangulation (implemented, waiting for camera poses)", h1),
      P("When the cameras' positions are known (extrinsics: rotation R and translation t of each camera, "
        "X<sub>cam</sub> = R·X<sub>world</sub> + t), a joint seen in several cameras is reconstructed by the "
        "<b>Direct Linear Transform (DLT)</b>. Each detected keypoint is first undistorted to normalised "
        "coordinates (x, y) with the lens calibration. With P = [R | t] the projection gives two linear equations "
        "per camera in the unknown homogeneous point X:"),
      P("w · (x · P<sub>3</sub> − P<sub>1</sub>) · X = 0,   w · (y · P<sub>3</sub> − P<sub>2</sub>) · X = 0",
        ParagraphStyle("eq2", parent=body, alignment=TA_CENTER, fontSize=10.5)),
      P("where P<sub>i</sub> is the i-th row of P and w the detection confidence (weight). Stacking all cameras gives "
        "A·X = 0, solved by singular value decomposition. Robustness: if a camera's reprojection error for a joint "
        "exceeds a threshold (e.g. a mis-detection), every leave-one-camera-out solution is computed and the most "
        "consistent one is kept; joints seen by fewer than two cameras are left empty."),
      P("Tests on a synthetic 3-camera fisheye rig: noise-free points are recovered to &lt; 1 mm; with 0.5 px noise and "
        "80 px errors injected into one camera for 10 points, the bad camera is rejected (one geometrically ambiguous "
        "case, where the error lies along an epipolar line, cannot be detected with only three cameras - temporal "
        "filtering will handle such cases)."),
      P("6. Software and reproducibility", h1),
      *B(["Python package <font name='DVM'>moppose</font>, Python 3.12, managed with uv; one command-line tool with "
          "sub-commands (probe, board-check, calib-intrinsics, people, pose2d, pose-preview, pose-all, setup)",
          "16 automated tests with synthetic data: a ChArUco board rendered through a known fisheye camera by ray "
          "casting (calibration must recover it), the 0.5 px correction, triangulation with missing/bad views, the "
          "virtual-camera round trip on the real cam1 calibration, and the mopper selection",
          "videos stay local; calibration results are versioned in git; every step writes visual reports so results "
          "can be checked by eye",
          "configuration in two small files: the board (board.yaml) and the session (which video belongs to which camera)"]),
      Spacer(1, 4),
      P("7. Next steps", h1),
      table([["step", "what is needed", "method"],
             ["Extrinsic calibration", "a reference object seen by all three cameras at the same time (e.g. markers "
                                       "or tape marks on the floor with measured positions)",
              "mark the same points in each camera; solve each camera's pose (PnP) in one world frame (metres, Z up); "
              "check by triangulating the points back"],
             ["Time synchronisation", "a shared event (e.g. clap, light switch) or the motion itself",
              "estimate the offset between recordings, refine by cross-correlating the 2D joint trajectories"],
             ["3D landmarks", "results of the two steps above", "triangulate the 23 joints per frame, reject outliers, "
                                                              "fill short gaps, smooth (low-pass filter)"],
             ["Joint angles", "3D landmarks", "fit a skeleton with constant bone lengths; output angles for the robot"]],
            [3.2 * cm, 6 * cm, W - 9.2 * cm]),
      Spacer(1, 8),
      P("<b>Known limitations.</b> Calibration is extrapolated in the image corners (ceiling/walls, outside the work "
        "area). With three cameras, a joint occluded in two views cannot be triangulated in that frame. Left/right "
        "confusion of limbs is possible when the person is seen from behind; multi-view consistency will be used to "
        "detect it."),
      PageBreak()]

# ---------------------------------------------------------------------------- appendix
s += [P("Appendix A - Calibration parameters (1280 x 1024, OpenCV fisheye model)", h1),
      table([["camera", "f<sub>x</sub>, f<sub>y</sub> [px]", "c<sub>x</sub>, c<sub>y</sub> [px]",
              "k<sub>1</sub>, k<sub>2</sub>, k<sub>3</sub>, k<sub>4</sub>", "RMS"],
             ["cam1", "911.43, 916.69", "595.69, 490.38", "-0.0967, -0.0368, -0.0092, 0.0669", "0.84 px"],
             ["cam3", "931.55, 939.22", "619.87, 540.50", "-0.1114, 0.0846, -0.1493, 0.1057", "0.95 px"],
             ["cam4", "901.87, 907.44", "635.95, 505.31", "-0.1276, 0.1752, -0.5372, 0.5481", "0.76 px"]],
            [1.6 * cm, 3.1 * cm, 3.1 * cm, W - 10.4 * cm, 2.6 * cm]),
      Spacer(1, 10),
      P("Appendix B - Main commands", h1),
      P("uv sync --all-extras            # install (PyTorch CUDA, models' dependencies)<br/>"
        "uv run moppose setup            # download Sapiens2, RTMW, YOLO26; GPU check<br/>"
        "uv run moppose probe &lt;videos&gt;   # real frame rate, dropped frames, contact sheet<br/>"
        "uv run moppose board-check &lt;video&gt;    # does board.yaml match the board?<br/>"
        "uv run moppose calib-intrinsics --cam cam1   # lens calibration + report<br/>"
        "uv run moppose people --cam cam1            # tracking, mopper selection<br/>"
        "uv run moppose pose2d --cam cam1            # 2D pose (Sapiens2 + RTMW)<br/>"
        "uv run moppose pose-all                     # everything for all cameras", mono),
      Spacer(1, 6),
      P("Appendix C - Calibration coverage of the other cameras", h1),
      fig("cam3_coverage.jpg", 8.2, "cam3: board points (green) and coverage heat map; red cells = no board points."),
      fig("cam4_coverage.jpg", 8.2, "cam4: same report."),
      ]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                        bottomMargin=1.8 * cm, title="Multi-camera 3D pose of a mopping person - progress report",
                        author="moppose project")
doc.build(s, onFirstPage=on_page, onLaterPages=on_page)
print("wrote", OUT, f"{OUT.stat().st_size / 1e6:.1f} MB")
