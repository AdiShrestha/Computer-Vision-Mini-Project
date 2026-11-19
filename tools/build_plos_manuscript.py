#!/usr/bin/env python3
"""Build the PLOS ONE submission (Manuscript.docx, SI tables, figures, ledger) from the result packet.

Every number is printed through the ledger (packet path, code constant, or a
documented fixed protocol value) and every directional statement is an
asserted claim; tools/verify_plos_package.py re-checks them against data.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "source/runners"))
from plos_docx import Manuscript, bar, d, f, hat, nary, r, sub, subsup, sup, t, tilde  # noqa: E402
from plos_ledger import Ledger  # noqa: E402

PACKET = ROOT / "docs/results/sr1_plos_packet.json"
REFS = ROOT / "docs/manuscript/references_plos.json"
FIGS = ROOT / "docs/figures/plos"
SUB = ROOT / "submission/plos_one"
OUT_ANALYSIS = ROOT / "data/derived/sr1_multiyear/attempt_007/plos_001"
SL, KT = "PILOT-SOUTH-LHONAK", "PILOT-KHANGCHUNG-TSHO"
AMEND = "docs/protocol/SR1_AMENDMENT_PLOS_001.json"
PARENT = "docs/protocol/SR1_PROTOCOL_V4_PARENT.json"

TITLE = ("Masked transformer reconstruction versus seasonal climatology for Sentinel-2 monitoring of two Sikkim "
         "glacial lakes around the 2023 South Lhonak outburst: a retrospective case study")
SHORT_TITLE = "Masked transformer vs climatology at two Sikkim glacial lakes"
AUTHOR = "Aditya Shrestha^1*^"
AFFIL = "^1^ Department of Computer Science and Engineering, School of Engineering, Kathmandu University, Dhulikhel, Kavre, Nepal"
CORRESP = "* as03233622@student.ku.edu.np"
NOTICE = "Contains modified Copernicus Sentinel data [2018–2023]."


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Builder:
    def __init__(self):
        self.packet = json.loads(PACKET.read_text())
        self.refs = json.loads(REFS.read_text())["references"]
        self.L = Ledger(self.packet)
        self.order: list[str] = []
        self.fixed: set[str] = set()
        self.m = Manuscript()

    # ---------------------------------------------------------------- helpers
    def n(self, path, spec=".3f", scale=1.0):
        return self.L.num(path, spec, scale)

    def c(self, module, attr, spec="g", scale=1.0):
        return self.L.const(module, attr, spec, scale)

    def fx(self, text, source):
        self.L.entries.append({"kind": "fixed", "text": text, "source": source})
        self.fixed.add(text)
        return text

    def claim(self, expr, desc):
        if not self.L.claim(expr, desc):
            raise SystemExit(f"manuscript claim does not hold for this packet: {desc} ({expr})")

    def cite(self, *keys):
        nums = []
        for k in keys:
            if k not in self.refs:
                raise KeyError(f"unverified reference {k}")
            if k not in self.order:
                self.order.append(k)
            nums.append(self.order.index(k) + 1)
        nums = sorted(set(nums))
        parts, i = [], 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            parts.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ",".join(str(x) for x in nums[i:j + 1]))
            i = j + 1
        return "[" + ",".join(parts) + "]"

    def P(self, text):
        return self.m.para(text)

    def H(self, text, level=1):
        return self.m.heading(text, level)

    def site(self, path_tmpl, site, spec=".3f"):
        return self.n(path_tmpl.replace("SITE", site), spec)

    # ---------------------------------------------------------------- shared short-hands
    def rc(self, role, name, spec=".3f"):
        return self.n(f"reconstruction.by_role.{role}.pooled_mse.{name}", spec)

    def diff(self, role, name):
        base = f"reconstruction.differences.{role}.{name}"
        return f"{self.n(base + '.estimate', 'signed3')} (95% CI {self.n(base + '.lower', 'signed3')} to {self.n(base + '.upper', 'signed3')})"

    def score(self, field, site, group, spec=".3f"):
        return self.n(f"scores.table[field={field}].{site}|{group}.median", spec)

    def pc(self, method, frac, key, spec="pct0"):
        return self.n(f"positive_control.methods.{method}.by_fraction['{frac}'].{key}", spec)

    # ---------------------------------------------------------------- document
    def build(self):
        m, n, c, fx = self.m, self.n, self.c, self.fx
        FEAT, SCOR, MOD, GEO = "sentinel_gl.sr1_features", "sentinel_gl.sr1_scoring", "sentinel_gl.sr1_model", "sentinel_gl.sr1_geometry"
        STATS, PCM, RUN, BASE = "sentinel_gl.sr1_stats", "sentinel_gl.sr1_positive_control", "run_sr1_plos_analysis", "sentinel_gl.sr1_baselines_true"

        # Fixed protocol values printed in the text, with their sources (labels, not measurements).
        for tok, src in [("10%", PCM + ".FRACTIONS (0.1)"), ("25%", PCM + ".FRACTIONS (0.25)"), ("50%", PCM + ".FRACTIONS (0.5)"),
                         ("95%", AMEND + " changes.C14_uncertainty"), ("95", AMEND + " changes.C13_positive_control (95th percentile)"),
                         ("0.0004", "sentinel_gl.sr1_features.candidate_water_area_baseline pixel_area_m2=400"),
                         ("1.4826", "sentinel_gl.sr1_features.fit_robust_scaler (normal-consistent MAD factor)"),
                         ("033", "Sentinel-2 relative orbit label in product URIs"),
                         ("05.00", "processing-baseline label"), ("05.09", "processing-baseline label"), ("05.10", "processing-baseline label")]:
            fx(tok, src)
        # Pre-declared rule outcomes that the wording depends on.
        self.claim("R('reconstruction.rule_R1_model_uses_context')", "context ablation: model uses its visible context (R1)")
        self.claim("not R('reconstruction.rule_R2_model_beats_seasonal_plus_offset.VAL')", "VAL interval vs seasonal+offset includes zero (R2 VAL)")
        self.claim("R('reconstruction.rule_R2_model_beats_seasonal_plus_offset.EVAL')", "EVAL interval vs seasonal+offset excludes zero (R2 EVAL)")
        self.claim("R('reconstruction.differences.VAL.patch_seasonal_mean_plus_visible_offset.estimate') < 0", "VAL point estimate favours the model")
        self.claim("R('reconstruction.differences.VAL.patch_seasonal_mean.upper') < 0", "model below per-patch seasonal mean on VAL")
        self.claim("R('reconstruction.differences.EVAL.patch_seasonal_mean.upper') < 0", "model below per-patch seasonal mean on EVAL")
        self.claim(f"R('scores.table[field=score_a_zone0_1].{SL}|EVAL.median') > R('scores.table[field=score_a_zone0_1].{SL}|VAL.median')", "SL EVAL Score A above SL VAL")
        self.claim(f"R('cusum.alarms.{SL}|VAL.scenes_majority') > 0", "SL CUSUM alarms before the event")
        self.claim(f"R('cusum.alarms.{KT}|EVAL.scenes_any_seed') == 0", "no KT EVAL CUSUM alarm")
        self.claim("R(\"positive_control.methods.score_a_zone0_1.by_fraction['0.1'].registers\")", "Score A registers 10% drainage")
        self.claim("not any(R(\"positive_control.methods.water_candidate_area_km2.by_fraction['%s'].registers\" % f) for f in ('0.1','0.25','0.5','1.0'))", "area threshold never crossed")
        self.claim("all(R(\"positive_control.methods.water_candidate_area_km2.by_fraction['%s'].median_paired_change\" % f) < 0 for f in ('0.1','0.25','0.5','1.0'))", "area decreases under perturbation")
        self.claim("not any(R(\"positive_control.methods.%s.by_fraction['%s'].registers\" % (m_, f)) for m_ in ('isolation_forest_score','one_class_svm_score','missingness_only_score') for f in ('0.1','0.25','0.5','1.0'))", "IF, OCSVM, missingness never register")
        self.claim("R(\"positive_control.methods.isolation_forest_score.by_fraction['1.0'].median_paired_change\") < 0 and R(\"positive_control.methods.one_class_svm_score.by_fraction['1.0'].median_paired_change\") < 0", "IF and OCSVM scores fall under perturbation")
        self.claim("R('replay.status') == 'PASS'", "independent replay passed")
        self.claim("all(R('reconstruction.by_role.%s.pooled_mse.model_all_context_hidden') > 2 * R('reconstruction.by_role.%s.pooled_mse.model') for _ in [0]) and R('reconstruction.by_role.VAL.pooled_mse.model_all_context_hidden') > 2 * R('reconstruction.by_role.VAL.pooled_mse.model') and R('reconstruction.by_role.EVAL.pooled_mse.model_all_context_hidden') > 2 * R('reconstruction.by_role.EVAL.pooled_mse.model')" % ('VAL', 'VAL'), "removing context more than doubles error (VAL and EVAL)")
        self.claim("all(R('reconstruction.by_role.%s.pooled_mse.calendar_site_mean') > 5 * R('reconstruction.by_role.%s.pooled_mse.patch_seasonal_mean') and R('reconstruction.by_role.%s.pooled_mse.fit_median') > 5 * R('reconstruction.by_role.%s.pooled_mse.patch_seasonal_mean') for _ in [0])" % ('FIT','FIT','FIT','FIT') + " and all(R('reconstruction.by_role.' + k + '.pooled_mse.calendar_site_mean') > 5 * R('reconstruction.by_role.' + k + '.pooled_mse.patch_seasonal_mean') for k in ['VAL','EVAL'])", "calendar-only and FIT-median an order of magnitude worse")
        self.claim(f"R('scores.eval_scenes[site_id={SL},date=2023-10-09].water_candidate_area_km2') < R('scores.table[field=water_candidate_area_km2].{SL}|AUTUMN_FIT_2018_2021.median')", "first SL EVAL area below autumn median")
        for s_ in (SL, KT):
            self.claim(f"max(R('scores.scene_score_a_max.{s_}|FIT'), R('scores.scene_score_a_max.{s_}|VAL')) > R('scores.scene_score_a_max.{s_}|EVAL')", f"pre-event maximum Score A exceeds post-event maximum at {s_}")
        self.claim(f"R('scores.table[field=score_a_zone_contrast].{SL}|EVAL.median') > R('scores.table[field=score_a_zone_contrast].{SL}|VAL.median')", "SL zone contrast above VAL")
        self.claim("R('training.fits_completed') == R('training.fits_total')", "no fit failed")

        # ------------------------------------------------ title page
        p = m.para(TITLE)
        p.runs[0].bold = True
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        m.para("Short title: " + SHORT_TITLE)
        m.para(AUTHOR)
        m.para(AFFIL)
        m.para(CORRESP)

        # ------------------------------------------------ abstract
        self.H("Abstract")
        s_ab = (
            "Cloud, snow and irregular revisits leave optical time series of Himalayan glacial lakes incomplete, and it is unclear "
            "whether learned spatiotemporal models add information beyond simple seasonal references. We assembled "
            f"{n('corpus.scenes_total', 'int')} source-bound Sentinel-2 Level-2A scenes (2018–2023) of South Lhonak, which drained in an outburst "
            "on 3–4 October 2023, and of Khangchung Tsho, with windows and analysis zones taken from an independent lake inventory. "
            f"A {n('training.cells[family=primary].parameter_count', 'int')}-parameter masked transformer with spatial and causal temporal attention "
            f"was trained with {n('training.cells[family=primary].completed', 'int')} seeds on 2018–2021 and early-2023 scenes, selected on 2022 and "
            "July–September 2023 scenes (validation) and applied to the post-event scenes. Every observed patch was reconstructed once "
            "and compared, on identical entries, with parameter-free seasonal predictors using moving-block bootstrap intervals. "
            f"The model used its context: hiding it raised validation error from {self.rc('VAL', 'model')} to {self.rc('VAL', 'model_all_context_hidden')}. "
            f"Its validation error was lower than that of a per-patch seasonal mean shifted by the scene's visible offset "
            f"({self.rc('VAL', 'patch_seasonal_mean_plus_visible_offset')}), but the 95% interval of the difference included zero; "
            f"after the event the difference was {self.diff('EVAL', 'patch_seasonal_mean_plus_visible_offset')}. "
            "In a constructed positive control replacing part of the lake with background reflectance, the reconstruction score exceeded its "
            f"training-period 95th percentile in {self.pc('score_a_zone0_1', '0.1', 'perturbed_detection_rate')} of held-out scenes at 10% "
            f"replacement (unperturbed: {self.pc('score_a_zone0_1', '0.1', 'clean_detection_rate')}), whereas Isolation Forest and one-class SVM "
            "did not respond. At South Lhonak the post-event reconstruction error "
            f"(median {self.score('score_a_zone0_1', SL, 'EVAL')}) exceeded that of held-out pre-event scenes ({self.score('score_a_zone0_1', SL, 'VAL')}), "
            f"but a cumulative-sum rule had already alarmed in {n(f'cusum.alarms.{SL}|VAL.scenes_majority', 'int')} of "
            f"{n(f'cusum.alarms.{SL}|VAL.scenes', 'int')} pre-event validation scenes. With two lakes, one event and no verified control lakes, "
            "detection timing, false-alert burden and warning value were not estimable. The model is a sensitive reconstruction "
            "reference, not a demonstrated warning tool.")
        self.P(s_ab)

        # ------------------------------------------------ introduction
        self.H("Introduction")
        self.P("Glacial lakes have grown in number and volume as glaciers retreat, and outburst floods from moraine- and ice-dammed lakes "
               f"threaten downstream communities across High Mountain Asia {self.cite('shugar2020', 'zheng2021', 'taylor2023')}. On 3–4 October 2023 "
               "South Lhonak Lake in Sikkim, India, partially drained, triggering a cascading flood along the Teesta River that destroyed the "
               f"Chungthang dam and caused many fatalities {self.cite('sattar2025')}. Satellite records have been used to reconstruct this event "
               f"and earlier Himalayan outbursts after the fact {self.cite('yu2024', 'veh2018')}.")
        self.P("Optical time series are the most direct satellite record of lake surfaces, but in the Himalaya they are interrupted by "
               f"cloud, seasonal snow and lake ice, and revisits are irregular even for the Sentinel-2 constellation {self.cite('drusch2012')}. "
               f"Water indices such as NDWI and MNDWI separate open water from land {self.cite('mcfeeters1996', 'xu2006')}, and seasonal-trend and "
               f"continuous change detection methods model expected behaviour from long records {self.cite('verbesselt2010', 'zhu2014', 'kennedy2010')}. "
               "Such methods are hard to apply when much of each scene is missing.")
        self.P("Masked modelling offers another route: a network learns to reconstruct hidden parts of its input from visible context "
               f"{self.cite('vaswani2017', 'he2022')}, and the idea has been carried over to multivariate time series "
               f"{self.cite('zerveas2021')} and to satellite image time series {self.cite('cong2022', 'tseng2023', 'szwarcman2024')}. Reconstruction "
               "error is then a natural anomaly score. Evaluations of time-series anomaly detectors, however, warn that benchmark design, "
               f"trivial baselines and event definitions often decide the outcome {self.cite('wu2021', 'tatbul2018', 'sarfraz2024')}.")
        self.P("We therefore asked four questions of a small masked spatiotemporal transformer trained on an incomplete Sentinel-2 record "
               "of two Sikkim lakes. (Q1) Does it reconstruct held-out lake patches better than parameter-free seasonal references evaluated "
               "on exactly the same entries? (Q2) Does it use the visible spatial and temporal context at all? (Q3) Do its anomaly scores "
               "register a known, constructed change? (Q4) What did the scores and simple comparators show around the 2023 outburst? "
               "The study is a retrospective case study with one event. It is exploratory: the 2023 scenes and an earlier, defective version "
               "of this analysis had been inspected before the analysis protocol reported here was frozen. Negative and inconclusive "
               "outcomes are reported with the same weight as positive ones.")

        # ------------------------------------------------ methods
        self.H("Materials and methods")
        self.H("Study sites and lake geometry", 2)
        gsl, gkt = f"geometry.sites.{SL}", f"geometry.sites.{KT}"
        self.P("South Lhonak (Sikkim, India) is the proglacial lake that drained on 3–4 October 2023. Khangchung Tsho, about 60 km to "
               "the east, is a proglacial lake used here as a development comparator; its outburst history has not been independently "
               "ascertained and it is not a verified control. Lake geometry was taken from the 2017 layer of the Hi-MAG inventory, "
               f"which provides annual 30 m Landsat-derived outlines for High Mountain Asia {self.cite('chen2021', 'chen2020data')}: outlines "
               f"GL088195E27912N ({n(gsl + '.hi_mag_area_km2', '.2f')} km², elevation {n(gsl + '.hi_mag_elevation_m', 'int')} m) and GL088815E27990N "
               f"({n(gkt + '.hi_mag_area_km2', '.2f')} km², {n(gkt + '.hi_mag_elevation_m', 'int')} m). The inventory is independent of our data and "
               "predates the study period; it is not a 2018–2023 shoreline.")
        self.P(f"Each site is analysed in a window of {fx('160', PARENT + ' scope.roi.shape')} × 160 pixels at {c(GEO, 'PIXEL_M', 'g')} m "
               f"in UTM zone 45N (EPSG:32645). The Khangchung Tsho window, centred on the study reference coordinate, contains "
               f"{n(gkt + '.outline_fraction_inside_new_window', 'pct0')} of the inventory outline. The South Lhonak window used in earlier "
               f"work contained only {n(gsl + '.outline_fraction_inside_previous_window', 'pct0')} of the outline, so it was re-centred on the "
               f"outline centroid (a shift of {n(gsl + '.window_shift_east_m', 'int', -1.0)} m west and {n(gsl + '.window_shift_north_m', 'int', -1.0)} m south), "
               f"after which it contains {n(gsl + '.outline_fraction_inside_new_window', 'pct0')} (Fig 1A). Zone 0 consists of pixels whose centre "
               f"lies inside the outline, zone 1 of other pixels within {c(GEO, 'BUFFER_M', 'g')} m of zone 0 (shore, dam and moraine), and zone 2 "
               "of the rest of the window (Fig 1A and 1B).")
        self.H("Sentinel-2 corpus", 2)
        cp = "corpus"
        PB0, PB9, PB10 = "corpus.processing_baselines['05.00']", "corpus.processing_baselines['05.09']", "corpus.processing_baselines['05.10']"
        ORB = f"corpus.relative_orbits.{SL}['033']"
        self.P("Bottom-of-atmosphere reflectance in bands B03 (green), B08 (near infrared) and B11 (shortwave infrared) and the scene "
               "classification layer (SCL) were read from Sentinel-2 Collection-1 Level-2A products through the Earth Search STAC API "
               f"{self.cite('earthsearch')} with HTTP range requests on cloud-optimised GeoTIFFs. For 2018–2022, one product was selected per "
               f"site in each of {fx('12', PARENT + ' scope.provider_selection')} equal-length periods of the year: the product with the lowest "
               "provider cloud-cover value among products fully covering the window (ties broken by time and identifier); no score or "
               "outcome was used. The 2023 scenes come from an earlier acquisition that applied the same rule with "
               f"{fx('24', 'source/sentinel_gl/ingestion/expansion2023.py select_optical(count=24)')} equal periods between 1 January and "
               "15 November 2023, so 2023 is sampled about twice as densely as earlier years. In 2022 the catalogue held only "
               f"{n(cp + f'.catalogue_inventory[site_id={SL},year=2022].eligible', 'int')} and "
               f"{n(cp + f'.catalogue_inventory[site_id={KT},year=2022].eligible', 'int')} eligible products for the two sites, "
               f"against {n(cp + f'.catalogue_inventory[site_id={SL},year=2021].eligible', 'int')} and "
               f"{n(cp + f'.catalogue_inventory[site_id={KT},year=2021].eligible', 'int')} in 2021, and the selection kept "
               f"{n(cp + f'.by_site_year.{SL}.2022', 'int')} scenes per site (March and December).")
        self.P(f"The corpus has {n(cp + '.scenes_total', 'int')} scenes ({n(cp + f'.by_site_role.{SL}.FIT', 'int', 1)}+{n(cp + f'.by_site_role.{SL}.VAL', 'int')}+{n(cp + f'.by_site_role.{SL}.EVAL', 'int')} "
               f"at South Lhonak and {n(cp + f'.by_site_role.{KT}.FIT', 'int')}+{n(cp + f'.by_site_role.{KT}.VAL', 'int')}+{n(cp + f'.by_site_role.{KT}.EVAL', 'int')} at Khangchung Tsho "
               f"in the three splits defined below; Table 1). Processing baselines are 05.00 ({n(PB0, 'int')} scenes), "
               f"05.09 ({n(PB9, 'int')}) and 05.10 ({n(PB10, 'int')}). "
               f"All Khangchung Tsho scenes come from relative orbit 033, which also supplies {n(ORB, 'int')} South Lhonak "
               f"scenes, and {n(cp + '.shared_acquisition_dates', 'int')} acquisition dates are common to both sites, so the sites are not "
               "independent replicates. Every scene is bound by SHA-256 digests to its STAC item, byte-range receipts and native SAFE "
               "metadata, and the loader refuses any scene whose bytes, time or origin do not match. Copernicus Sentinel data are "
               f"provided under a free, full and open data policy {self.cite('copernicus_notice')}.")
        self.H("Preprocessing and patch features", 2)
        self.P("Digital numbers were converted to reflectance with each product's native quantification value and additive offset; "
               "valid negative reflectance was kept. The 10 m bands were averaged to 20 m over 2 × 2 blocks in which all four pixels "
               "were valid. Pixels outside SCL classes 4, 5 and 6 (vegetation, not vegetated, water) were set to missing, so snow, ice, "
               f"cloud and shadow are treated as unobserved. Each window was divided into {c(SCOR, 'GRID_SIDE', 'g')} × 10 patches of "
               f"{c(FEAT, 'PATCH_SIZE', 'g')} × 16 pixels; a patch is observed when at least {c(FEAT, 'MIN_PATCH_SUPPORT', 'pct0')} of its "
               "pixels are clear. Its features are the mean B03, B08 and B11 reflectance over clear pixels and NDWI and MNDWI computed "
               "from those means (Eq 1), where an index is missing when its denominator is smaller in magnitude than "
               f"δ = {c(FEAT, 'DELTA_DENOMINATOR', 'g')}.")
        rho = lambda b: sub(bar(r("ρ")), t(b))  # noqa: E731
        m.equation(r("NDWI") + t(" = ") + f(rho("03") + t(" − ") + rho("08"), rho("03") + t(" + ") + rho("08")) + t(",   ") +
                   r("MNDWI") + t(" = ") + f(rho("03") + t(" − ") + rho("11"), rho("03") + t(" + ") + rho("11")), 1)
        self.P("Features were standardized per channel with the median and 1.4826 × the median absolute deviation (MAD) of all FIT "
               "observations, pooled over both sites (Eq 2). Patch zone labels are 0 when at least "
               f"{c(FEAT, 'ZONE0_PATCH_FRACTION', 'pct0')} of the patch's pixels are in zone 0, otherwise 1 when at least "
               f"{c(FEAT, 'ZONE1_PATCH_FRACTION', 'pct0')} are in zone 1, otherwise 2; this gives "
               f"{n(f'geometry.patch_zone_counts.{SL}.0', 'int')}, {n(f'geometry.patch_zone_counts.{SL}.1', 'int')} and {n(f'geometry.patch_zone_counts.{SL}.2', 'int')} "
               f"patches in zones 0, 1 and 2 at South Lhonak and {n(f'geometry.patch_zone_counts.{KT}.0', 'int')}, {n(f'geometry.patch_zone_counts.{KT}.1', 'int')} "
               f"and {n(f'geometry.patch_zone_counts.{KT}.2', 'int')} at Khangchung Tsho. Time is encoded by the sine and cosine of day of year "
               "and by the gap since the previous scene, min(Δt / 30 days, 5); no monotonic time is used.")
        m.equation(sub(tilde(r("x")), r("t,p,c")) + t(" = ") + f(sub(r("x"), r("t,p,c")) + t(" − ") + sub(t("median"), r("c")),
                                                                  t("1.4826 ") + sub(t("MAD"), r("c"))), 2)
        self.H("Splits", 2)
        self.P("Scenes are assigned by acquisition time. FIT (model fitting) comprises 2018–2021 and January–June 2023; VAL (checkpoint "
               "selection) comprises 2022 and July–September 2023; scenes from 2 October 2023 18:30 to 4 October 2023 18:30 UTC (3–4 October "
               "in Sikkim) are excluded as event-uncertain; EVAL comprises later scenes (to 15 November 2023). The split is retrospective, not "
               "forward in time: VAL 2022 lies between FIT years, and 2018–2021 autumn scenes used for comparison below were used to fit "
               "the model. Scaling statistics, PCA, reference banks and score moments use FIT scenes only; VAL scenes never enter an "
               "optimizer step; no fitted state uses EVAL.")
        self.H("Model", 2)
        self.P(f"The model projects each observed patch's {fx('5', AMEND + ' unchanged_from_parent')} "
               f"features to a {c(MOD, 'SR1TransformerMAE:d_model', 'g')}-dimensional token and adds learned patch-position and zone "
               "embeddings, a projection of the time features and a constant modality embedding (a second, unused modality row accounts "
               f"for 64 of the trainable parameters). Two Transformer encoder layers with {c(MOD, 'SR1TransformerMAE:nhead', 'g')} heads, "
               f"feed-forward width {c(MOD, 'SR1TransformerMAE:dim_feedforward', 'g')}, dropout {c(MOD, 'SR1TransformerMAE:dropout', 'g')} and GELU "
               "activation attend across observed patches within a scene; one causal encoder layer then attends, for each patch location "
               "separately, only to that location's earlier observed times; one spatial decoder layer and a linear head reconstruct the "
               "features. Unobserved patches are excluded from attention, and hidden (masked) patches enter as a learned mask token. The "
               f"model has {n('training.cells[family=primary].parameter_count', 'int')} trainable parameters ({n('training.cells[family=capacity,cell=width_32].parameter_count', 'int')} "
               f"and {n('training.cells[family=capacity,cell=width_128].parameter_count', 'int')} for the capacity variants with d = 32 and 128). "
               f"Inputs are sequences of up to {c(SCOR, 'MAX_CONTEXT', 'g')} consecutive scenes of one site.")
        self.H("Training", 2)
        self.P("In every scene of a training sequence with n ≥ 2 observed patches, round(n/2) of them are hidden, drawn at random, and "
               "at least one stays visible; unobserved patches are never targets. The loss is the mean squared error over the observed "
               f"channels of hidden patches. Optimisation used AdamW {self.cite('loshchilov2019')} with learning rate "
               f"{c(MOD, 'train_factorized_sr1_model:lr', 'g')}, weight decay {c(MOD, 'train_factorized_sr1_model:weight_decay', 'g')}, cosine "
               f"annealing to {c(MOD, 'train_factorized_sr1_model:min_lr', '.5f')}, gradient clipping at norm 1, one sequence per step, and an "
               f"exponential moving average of the weights (decay {c(MOD, 'EMA:decay', 'g')}), in PyTorch {self.cite('paszke2019')}. Each "
               "epoch, VAL scenes were reconstructed with every other observed patch hidden; the selection metric is the sum of squared "
               "errors over all VAL target entries divided by their number, so that a sequence holding one sparse scene does not carry "
               "extra weight. The checkpoint is the averaged-weight state at the epoch with the lowest VAL error, from at most "
               f"{c(MOD, 'train_factorized_sr1_model:max_epochs', 'g')} epochs, stopping after {c(MOD, 'train_factorized_sr1_model:plateau_patience', 'g')} "
               "epochs without improvement.")
        self.P(f"The primary configuration (5 channels, inventory zones, context 8, d = 64) was trained with {n('training.cells[family=primary].completed', 'int')} seeds. "
               "Capacity (d = 32 and 128) and a 2 × 2 × 2 factorial of channels (3 reflectance channels or all 5), zones (inventory or a "
               "constant label) and context (1 or 8 scenes) used 5 seeds per cell. Every primary fit that stopped at the epoch cap was "
               f"refitted with a {fx('450', AMEND + ' changes.C5_seeds_and_budget')}-epoch budget as an extended-budget check; the "
               "150-epoch checkpoints remain the primary models. Because the cells differ in their targets, each cell's VAL error is "
               "shown beside the per-patch seasonal FIT mean evaluated on the same VAL entries. Training ran on CPU "
               f"({n('training.workers', 'int')} worker processes with {n('training.torch_threads_per_worker', 'int')} threads each).")
        self.H("Reconstruction comparison", 2)
        self.P("For evaluation, the patches are split into two fixed checkerboard folds. For each target scene, the observed patches of "
               "one fold are hidden while the other fold and up to seven earlier scenes stay visible, then the folds swap, so each observed "
               "patch is hidden exactly once and the folds are the same for every seed. The same hidden entries are predicted by five "
               "parameter-free references built from FIT scenes of the same site, excluding the target scene itself: the FIT median "
               "(zero after standardization); the per-patch FIT mean; the per-patch seasonal mean over FIT scenes within "
               f"±{c(SCOR, 'SEASON_WINDOW_DAYS', 'g')} days of day of year; that seasonal mean plus the mean difference, per channel, between "
               "the target scene's visible fold and the seasonal mean (a 'visible offset' that uses the same spatial context the model "
               "sees); and a calendar-only site mean (all patches within ±45 days, no patch identity). Where a seasonal reference is "
               "unavailable the per-patch mean, then the calendar mean, is used. Two ablations reconstruct the same entries with the same "
               "checkpoints after setting every visible input to zero, and after hiding every observed patch of every context scene. "
               "Errors are pooled over entries (sum of squared errors divided by the number of entries) per split; model errors are "
               "averaged over seeds entry by entry.")
        self.H("Anomaly scores and comparators", 2)
        self.P("Score A is the reconstruction error of hidden patches averaged within zones 0 and 1 (Eq 3), where Ω is the set of "
               "observed zone 0 and 1 patches and C the observed channels of patch p; the same quantity over zone 2 gives a background "
               "value and the difference is the zone contrast. Score B (Eq 4) measures how far each observed zone 0 or 1 patch's latent "
               f"vector, after a rank-{c(SCOR, 'PCA_RANK', 'g')} principal component projection fitted to the latent vectors of observed FIT "
               f"patches, lies from the same patch in its {c(SCOR, 'SCORE_B_K', 'g')} nearest FIT scenes of the same site within ±45 days, "
               "using only references observed at that patch; P is the set of eligible patches. Score C (Eq 5) combines the two with "
               "robust standardization whose median and MAD are fitted on the FIT scenes of each site separately. A score is missing, "
               "with a recorded reason, when its support is empty.")
        sq = sup(d(sub(tilde(r("x")), r("t,p,c")) + t(" − ") + sub(hat(r("x")), r("t,p,c"))), t("2"))
        inner = f(t("1"), d(sub(r("C"), r("p")), "|", "|")) + nary("∑", r("c∈") + sub(r("C"), r("p")), sq)
        m.equation(sub(r("A"), r("t")) + t(" = ") + f(t("1"), d(r("Ω"), "|", "|")) + nary("∑", r("p∈Ω"), inner), 3)
        m.equation(sub(r("B"), r("t")) + t(" = ") + f(t("1"), d(r("P"), "|", "|")) +
                   nary("∑", r("p∈P"), f(t("1"), t("5")) + nary("∑", r("i∈") + sub(r("N"), t("5")) + d(r("t,p")),
                                                                d(sub(r("z"), r("t,p")) + t(" − ") + sub(r("z"), r("i,p")), "‖", "‖"))), 4)
        m.equation(sub(r("C"), r("t")) + t(" = 0.5 ") + sub(r("z"), t("FIT")) + d(t("ln ") + sub(r("A"), r("t"))) + t(" + 0.5 ") +
                   sub(r("z"), t("FIT")) + d(sub(r("B"), r("t"))) + t(",   ") + sub(r("z"), t("FIT")) + d(r("u")) + t(" = ") +
                   f(r("u") + t(" − median"), t("1.4826 MAD")), 5)
        self.P("A one-sided Page cumulative sum (CUSUM) is run on Score C for each seed and site (Eq 6) with "
               f"κ = {c(BASE, 'add_page_cusum:kappa', 'g')} and h = {c(BASE, 'add_page_cusum:threshold', 'g')}, without resets "
               f"{self.cite('page1954')}; a crossing of h is recorded as an exploratory alarm. These values were fixed in advance and not "
               "calibrated to any false-alarm rate.")
        m.equation(sub(r("S"), r("t")) + t(" = max") + d(t("0, ") + sub(r("S"), r("t−1")) + t(" + ") + sub(r("C"), r("t")) + t(" − ") + r("κ")) +
                   t(",   alarm if ") + sub(r("S"), r("t")) + t(" ≥ ") + r("h"), 6)
        self.P("Comparators fitted on complete five-channel zone 0 and 1 FIT patch vectors are: a seasonal residual (median distance "
               "to the five closest same-patch FIT vectors within ±45 days); temporal persistence (distance to the previous complete "
               f"vector at the same patch); Isolation Forest {self.cite('liu2008')} ({fx('100', BASE + ' fit_descriptive_baselines')} trees, contamination "
               f"{fx('0.05', BASE + ' fit_descriptive_baselines')}, at most {fx('4,000', BASE + ' fit_descriptive_baselines cap')} evenly spaced FIT vectors) and a "
               f"one-class SVM {self.cite('scholkopf2001')} (RBF kernel, ν = 0.05), both in scikit-learn {self.cite('pedregosa2011')}; and a "
               "missingness-only score, the fraction of zone 0 and 1 patches without a complete observation. A candidate water area counts "
               "clear zone 0 and 1 pixels with NDWI > 0 and MNDWI > 0 (0.0004 km² per pixel); it is a spectral diagnostic, not a "
               "validated shoreline, and seasonal snow and ice drive it towards zero. Full-rank robust PCA "
               f"{self.cite('candes2011')}, used in earlier versions, was removed because on five channels its residual is zero by construction.")
        self.H("Constructed positive control", 2)
        self.P("To test whether the scores can register a known change at all, each VAL and EVAL scene was perturbed by replacing the "
               "clear pixels of the easternmost fraction f of the inventory lake pixels (the dam end) with the scene's own median clear "
               f"background (zone 2) reflectance, labelled as bare ground, for f = {', '.join(fx(f'{v:g}', PCM + '.FRACTIONS') for v in (0.1, 0.25, 0.5))} "
               f"and {fx('1', PCM + '.FRACTIONS')}. Each perturbed scene was scored in its real context by the frozen checkpoints, PCA, "
               "moments and comparators. A scene counts as detected when its seed-median value exceeds the 95th percentile of that "
               "method's scene-level FIT values at the same site (for candidate area, falls below the 5th percentile), and the rate is "
               "compared with the rate for the same scenes unperturbed. Scenes without a clear lake pixel in the replaced part, or "
               "without clear background, were not perturbed. These arrays are constructed data and are labelled as such.")
        self.H("Statistical analysis", 2)
        self.P("Scenes of one lake are serially dependent, so differences in pooled error between the model and each reference or "
               f"ablation get 95% percentile intervals from a moving-block bootstrap {self.cite('kunsch1989')} over the chronologically "
               "ordered scenes of each site, resampled separately per site; the block length in scenes is the number spanning "
               f"{c(STATS, 'paired_ratio_difference:block_days', 'g')} days at the median revisit gap of the subset, and "
               f"{c(STATS, 'paired_ratio_difference:n_boot', 'int')} draws were used. Variation across seeds is reported as the sample standard "
               "deviation and describes optimisation noise on one corpus, not sampling uncertainty. Scene-level summaries take the median "
               "over seeds first, then medians and interquartile ranges over scenes. With three and four post-event scenes, no inferential "
               "comparison of post-event and earlier scores was attempted. Missing values are never imputed: each score, comparator and "
               "reference carries an explicit status, and summaries use the available scenes, whose numbers are reported.")
        self.H("Protocol, corrections and reproducibility", 2)
        self.P("The analysis follows a frozen parent protocol and an amendment (S1 File) whose SHA-256 digest and the digests of the "
               "training and scoring code were recorded before any model was trained. The amendment was written after a review of an "
               "earlier, unpublished version of this analysis found that its training mask hid every observed patch of every training "
               "scene, so the model never learned to use visible context; that its selection loss weighted sequences rather than targets; "
               "that one score was computed on unobserved patches; that one comparator was zero by construction; and that the South "
               "Lhonak window truncated the lake. All are corrected here; regression tests that failed on the earlier code are part of the "
               "code release, and no earlier result is reused. Three reporting additions were made after the results were seen "
               "(a VAL-only column in Table 4, a count of autumn comparisons in which the post-event median exceeded the autumn FIT median, "
               "and the largest scene-level Score A per site and split, used in one Discussion sentence), together with one cosmetic change "
               "to a script\u2019s status output; none alters a checkpoint, prediction, score or pre-declared threshold (S1 File). "
               "Every reported number is generated from a result packet that is recomputed from "
               "the retained prediction rows by a verification script, and an independent implementation replays all "
               f"{n('training.cells[family=primary].completed', 'int')} primary checkpoints with exact agreement. Software: Python "
               f"{self.fx(self.packet['software']['python'], 'docs/results/sr1_plos_packet.json software.python')}, "
               f"PyTorch {self.fx(self.packet['software']['packages']['torch'], 'packet software')}, NumPy {self.fx(self.packet['software']['packages']['numpy'], 'packet software')}, "
               f"scikit-learn {self.fx(self.packet['software']['packages']['scikit-learn'], 'packet software')}, rasterio {self.fx(self.packet['software']['packages']['rasterio'], 'packet software')} "
               f"and matplotlib {self.fx(self.packet['software']['packages']['matplotlib'], 'packet software')}, with a hash-pinned dependency lock.")
        self.H("Use of AI tools", 2)
        self.P("The author conceived the study and its research questions, chose the study lakes and data sources, and designed "
               "the approach: source-bound acquisition of Sentinel-2 windows, missingness-aware masked spatiotemporal modelling, the "
               "time-based fitting, validation and post-event splits, and comparison with simple references. The author also "
               "commissioned the code reviews and the pre-submission review that identified the defects corrected here, approved the "
               "corrections, and interpreted the results. AI assistants were used extensively to carry out this work: OpenAI Codex and "
               "Google Antigravity agents to implement and run earlier pipeline stages and drafts, and Anthropic Claude (Claude Opus "
               "5.5 in Claude Code) to review code, implement the corrections and the re-analysis, produce the figures, check the "
               "references and draft and edit this manuscript under the author's direction. No AI tool produced observations, labels "
               "or result values; all values come from executed code on the retained data and are checked by the automated "
               "verification described above. The author reviewed the code and text and takes full responsibility for the content.")

        # ------------------------------------------------ results
        self.H("Results")
        self.H("Corpus and geometry", 2)
        self.P(f"Of the {n('corpus.scenes_total', 'int')} scenes, {n('corpus.zero_observed_patch_scenes_total', 'int')} had no observed patch "
               f"(Table 1); the median scene had {n('corpus.observed_patches_quartiles.median', 'int')} of 100 patches observed (interquartile "
               f"range {n('corpus.observed_patches_quartiles.q1', 'int')}–{n('corpus.observed_patches_quartiles.q3', 'int')}). Clear-sky fractions "
               "were lowest in the monsoon months (Fig 1C and 1D). The spectral water rule used in earlier versions (water in at least half "
               "of the supported FIT scenes) agreed with the inventory lake mask with an intersection over union of "
               f"{n(gsl + '.spectral_vs_inventory.iou', '.2f')} at South Lhonak and {n(gkt + '.spectral_vs_inventory.iou', '.2f')} at Khangchung "
               "Tsho. Both lakes extend beyond their 2017 outlines in places (Fig 1A and 1B), consistent with lake growth after 2017.")
        self.fig_caption(1, "Study windows and observation record.",
                         f"(A, B) Median FIT false-colour composite (B11, B08, B03) of each {self.fx('3.2', 'window side: 160 pixels x 20 m (docs/protocol/SR1_PROTOCOL_V4_PARENT.json scope.roi)')} km window with the Hi-MAG 2017 outline (zone 0) and "
                         "the 500 m buffer (zone 1); white pixels had no clear FIT observation; the dotted square in A is the earlier, truncating "
                         "South Lhonak window. Thin white lines mark the 10 × 10 patch grid. Coordinates are UTM zone 45N. (C, D) Clear-pixel "
                         "fraction of every scene by split; the dashed line marks the 3–4 October 2023 outburst. " + NOTICE)
        self.table1()
        self.H("Training", 2)
        self.P(f"All {n('training.fits_total', 'int')} fits completed. The {n('training.cells[family=primary].completed', 'int')} primary seeds "
               f"reached a selected VAL error of {n('training.cells[family=primary].mean_val_mse', '.3f')} (sample SD "
               f"{n('training.cells[family=primary].sample_sd_val_mse', '.3f')}), against {n('training.validation_reference_mse_by_channels.5.patch_seasonal_mean', '.3f')} "
               f"for the per-patch seasonal mean on the same entries (Fig 2A, Table 2; all fits in S1 Table). Of the primary fits, "
               f"{n('training.cells[family=primary].stopping.MAX_EPOCH_BUDGET_REACHED', 'int')} stopped at the epoch cap; with 450 epochs those seeds reached {n('training.cells[family=extended_budget].mean_val_mse', '.3f')} "
               f"on average (SD {n('training.cells[family=extended_budget].sample_sd_val_mse', '.3f')}), so the primary models are not fully converged. "
               "In the factorial, context-1 cells had lower VAL error than context-8 cells "
               f"(for example {n('training.cells[family=factorial,cell=channels_5_zones_inventory_context_1].mean_val_mse', '.3f')} for 5 channels with inventory "
               "zones), but context-1 fits take more optimizer steps per epoch, so this does not isolate the value of temporal context. "
               "Inventory zones gave lower error than a constant zone in every channel and context combination (Fig 2B).")
        for ch in (3, 5):
            for ctx in (1, 8):
                inv = "primary" if (ch, ctx) == (5, 8) else "factorial"
                inv_cell = "channels_5_zones_inventory_context_8" if inv == "primary" else f"channels_{ch}_zones_inventory_context_{ctx}"
                self.claim(f"R('training.cells[family={inv},cell={inv_cell}].mean_val_mse') < R('training.cells[family=factorial,cell=channels_{ch}_zones_constant_context_{ctx}].mean_val_mse')",
                           f"inventory zones better than constant at {ch} ch, context {ctx}")
        self.claim("R('training.cells[family=factorial,cell=channels_5_zones_inventory_context_1].mean_val_mse') < R('training.cells[family=primary].mean_val_mse')", "context 1 lower than context 8")
        self.fig_caption(2, "Training outcomes.",
                         "(A) Selected VAL error of the ten primary seeds (filled) and, for seeds that stopped at the 150-epoch cap, of the same "
                         "seed refitted with 450 epochs (open); dashed line: per-patch seasonal FIT mean on the same VAL entries. (B) Mean ± sample "
                         "SD over seeds for every capacity and factorial cell; dashed lines: the seasonal reference for 5 and 3 channels.")
        self.table2()
        self.H("Reconstruction compared with references (Q1, Q2)", 2)
        self.P(f"On VAL the model's pooled error was {self.rc('VAL', 'model')} (seed range {self.rc('VAL', 'model__seed_min')}–{self.rc('VAL', 'model__seed_max')}) "
               f"over {n('reconstruction.by_role.VAL.entries', 'int')} entries in {n('reconstruction.by_role.VAL.scenes', 'int')} scenes (Fig 3, Table 3). "
               f"It was lower than the per-patch seasonal mean ({self.rc('VAL', 'patch_seasonal_mean')}; difference {self.diff('VAL', 'patch_seasonal_mean')}) "
               f"and the per-patch mean ({self.rc('VAL', 'patch_mean')}). Against the seasonal mean with visible offset ({self.rc('VAL', 'patch_seasonal_mean_plus_visible_offset')}) "
               f"the difference was {self.diff('VAL', 'patch_seasonal_mean_plus_visible_offset')}: the interval includes zero, so on VAL a parameter-free "
               "reference using the same visible context reconstructs about as well. On EVAL the model's error was "
               f"{self.rc('EVAL', 'model')} against {self.rc('EVAL', 'patch_seasonal_mean_plus_visible_offset')} for that reference "
               f"(difference {self.diff('EVAL', 'patch_seasonal_mean_plus_visible_offset')}) and {self.rc('EVAL', 'patch_seasonal_mean')} for the "
               "seasonal mean. The calendar-only and FIT-median predictors were an order of magnitude worse in every split, showing that "
               "patch identity carries most of the predictable signal.")
        self.P(f"The model depends on its context. Setting the visible inputs to zero raised VAL error to {self.rc('VAL', 'model_visible_inputs_zeroed')} "
               f"(difference {self.diff('VAL', 'model_visible_inputs_zeroed')}) and hiding all context raised it to {self.rc('VAL', 'model_all_context_hidden')} "
               f"({self.diff('VAL', 'model_all_context_hidden')}); the corresponding EVAL values were {self.rc('EVAL', 'model_visible_inputs_zeroed')} and "
               f"{self.rc('EVAL', 'model_all_context_hidden')}. On FIT scenes, which the model was fitted to, its error was {self.rc('FIT', 'model')}.")
        self.fig_caption(3, "Reconstruction error of the model, its ablations and parameter-free references.",
                         "(A) Pooled mean squared error on identical hidden entries by split (log scale). (B) Model error minus comparator error "
                         "with 95% stratified moving-block bootstrap intervals; negative values favour the model. FIT-median and calendar-only "
                         "differences are in Table 3.")
        self.table3()
        self.H("Scores around the outburst (Q4)", 2)
        sl_ev = lambda date, fld="score_a_zone0_1", spec=".3g": self.n(f"scores.eval_scenes[site_id={SL},date={date}].{fld}", spec)  # noqa: E731
        self.P(f"At South Lhonak, Score A in the three post-event scenes was {sl_ev('2023-10-09')} (9 October), {sl_ev('2023-10-26')} (26 October) "
               f"and {sl_ev('2023-11-13')} (13 November), against medians of {self.score('score_a_zone0_1', SL, 'VAL')} in the held-out pre-event VAL "
               f"scenes and {self.score('score_a_zone0_1', SL, 'AUTUMN_FIT_2018_2021')} in the October–November FIT scenes of 2018–2021 (Table 4, Fig 4; "
               "all scenes in S2 Table). "
               f"The zone contrast rose to {self.score('score_a_zone_contrast', SL, 'EVAL')} from {self.score('score_a_zone_contrast', SL, 'VAL')} in VAL. "
               f"Candidate water area in the first post-event scene was {sl_ev('2023-10-09', 'water_candidate_area_km2', '.2f')} km², against a "
               f"median of {self.score('water_candidate_area_km2', SL, 'AUTUMN_FIT_2018_2021', '.2f')} km² in earlier autumns, and recovered to "
               f"{sl_ev('2023-11-13', 'water_candidate_area_km2', '.2f')} km² by 13 November. At Khangchung Tsho, post-event Score A "
               f"({self.score('score_a_zone0_1', KT, 'EVAL')}) was close to its VAL ({self.score('score_a_zone0_1', KT, 'VAL')}) and autumn FIT "
               f"({self.score('score_a_zone0_1', KT, 'AUTUMN_FIT_2018_2021')}) medians. Pre-event scores were also variable: the largest Score A values "
               "at both sites came from 2023 pre-event scenes, two of them with a single observed patch and one a July monsoon scene "
               "(Fig 4A and 4B).")
        self.P(f"The CUSUM on Score C did not separate the event (Table 5). At South Lhonak it crossed h in all {n(f'cusum.alarms.{SL}|EVAL.scenes', 'int')} "
               f"post-event scenes for a majority of seeds, but it had already crossed in {n(f'cusum.alarms.{SL}|VAL.scenes_majority', 'int')} of "
               f"{n(f'cusum.alarms.{SL}|VAL.scenes', 'int')} VAL scenes (first on {self.fx(self.packet['cusum']['alarms'][SL + '|VAL']['first_majority'], 'packet cusum.alarms first_majority')}) "
               f"and, for at least one seed, in {n(f'cusum.alarms.{SL}|FIT.scenes_any_seed', 'int')} of {n(f'cusum.alarms.{SL}|FIT.scenes', 'int')} FIT scenes "
               f"from {self.fx(self.packet['cusum']['alarms'][SL + '|FIT']['first_any_seed'], 'packet cusum.alarms first_any_seed')}. At Khangchung Tsho no post-event "
               f"scene alarmed, while {n(f'cusum.alarms.{KT}|FIT.scenes_any_seed', 'int')} FIT and {n(f'cusum.alarms.{KT}|VAL.scenes_any_seed', 'int')} VAL "
               "scenes did for at least one seed. The comparators showed a higher seasonal residual at South Lhonak after the event "
               f"({self.score('baselines.seasonal_climatology_residual', SL, 'EVAL')} against {self.score('baselines.seasonal_climatology_residual', SL, 'VAL')} in VAL), "
               "while Isolation Forest and one-class SVM scores did not change appreciably (Table 4).")
        self.fig_caption(4, "Scores through time.",
                         "Score A (zone 0+1, log scale), Score C, the CUSUM statistic on Score C (dashed horizontal line: h = 4) and candidate water "
                         "area for South Lhonak (A, C, E, G) and Khangchung Tsho (B, D, F, H). Points are medians over the ten primary seeds and "
                         "vertical bars the seed range; lines are broken at gaps longer than 60 days; the dashed vertical line marks 3–4 October 2023. "
                         + NOTICE)
        self.table4()
        self.table5()
        self.H("Positive control (Q3)", 2)
        pcn = lambda meth, frac, key, spec="pct0": self.pc(meth, frac, key, spec)  # noqa: E731
        self.P(f"Perturbation was applicable to between {pcn('score_a_zone0_1', '0.1', 'scenes', 'int')} and {pcn('score_a_zone0_1', '1.0', 'scenes', 'int')} "
               "VAL and EVAL scenes; the others had no clear lake pixel in the replaced part or no clear background. Score A "
               f"exceeded its FIT threshold in {pcn('score_a_zone0_1', '0.1', 'perturbed_detection_rate')} of perturbed scenes already at 10% "
               f"replacement, against {pcn('score_a_zone0_1', '0.1', 'clean_detection_rate')} for the same scenes unperturbed (Fig 5, Table 6, S3 Table). "
               f"Score C reached {pcn('score_c', '0.25', 'perturbed_detection_rate')} at 25% and {pcn('score_c', '0.5', 'perturbed_detection_rate')} at 50%; "
               f"the seasonal residual {pcn('seasonal_climatology_residual', '0.5', 'perturbed_detection_rate')} and temporal persistence "
               f"{pcn('spatial_persistence_residual', '0.5', 'perturbed_detection_rate')} at 50%; Score B only {pcn('score_b_zone0_1', '1.0', 'perturbed_detection_rate')} "
               "at full replacement. Candidate water area fell as expected (median change "
               f"{pcn('water_candidate_area_km2', '1.0', 'median_paired_change', 'signed2')} km² at full replacement) but never below its FIT 5th "
               "percentile, which is near zero because winter snow and ice cover the lake in many FIT scenes. Isolation Forest and the "
               "one-class SVM never exceeded their unperturbed rates; their scores fell, that is, the replaced pixels looked more typical "
               "to them. The missingness-only score did not change, as intended. Clean-scene rates for Score A were high because FIT "
               "scenes, which set the threshold, are in-sample for the model.")
        self.fig_caption(5, "Constructed positive control.",
                         "Share of VAL and EVAL scenes whose seed-median value exceeds the method's FIT 95th percentile (for candidate water area, "
                         "falls below the FIT 5th percentile) for perturbed scenes (solid) and the same scenes unperturbed (dashed), by the "
                         "fraction of lake pixels replaced with background reflectance. n is the number of scenes that could be perturbed.")
        self.table6()
        self.H("Estimands", 2)
        self.P("Of the five quantities the protocol defined, four were not estimable and one is descriptive only (Table 7).")
        self.table7()

        # ------------------------------------------------ discussion
        self.H("Discussion")
        self.P("With its training corrected, the masked transformer uses visible context, and removing that context more than doubles its "
               "reconstruction error. Against parameter-free references evaluated on identical entries, it was clearly better than "
               "per-patch and seasonal means, but on the held-out pre-event scenes it was not distinguishable from a seasonal mean shifted "
               "by the scene's visible offset; after the event, with seven scenes, it was modestly better. Most of the predictable signal "
               "is carried by patch identity and season, which simple predictors capture. Studies of learned detectors for lake hazards "
               "should therefore report such references alongside the model, as recommended for time-series anomaly detection generally "
               f"{self.cite('wu2021', 'sarfraz2024')}.")
        self.P("The positive control shows that reconstruction error is sensitive: replacing a tenth of the lake with background "
               "reflectance moved almost every scene above the training-period threshold. Density-based comparators trained on pooled "
               "patch vectors did not respond, and the latent-distance score responded only to complete replacement. The candidate water "
               "area responded in the expected direction but a fixed percentile threshold over all seasons could not use it; a seasonal "
               "or event-window reference would be needed. Sensitivity is not specificity: individual pre-event 2023 scenes, among them "
               "a July monsoon scene and scenes with a single observed patch, had higher Score A than any post-event scene at the same "
               "lake (Fig 4A and 4B), and the CUSUM accumulated such values into alarms months before the event.")
        self.P("At South Lhonak, the post-event scenes had higher reconstruction error and zone contrast than the held-out pre-event "
               "scenes, and a lower candidate water area in the first scene, while Khangchung Tsho changed little. This is the pattern "
               "expected from partial drainage, but it rests on three scenes at one lake, and the earlier inspection of the 2023 data means "
               "it cannot be treated as a confirmatory test. Two of the post-event scenes are also more than three weeks after the event, "
               "when the surface had partly recovered.")
        self.H("Limitations", 2)
        self.P("The study has two lakes and one event, and Khangchung Tsho is a development comparator, not a verified event-free control; "
               "the sites share an orbit and many acquisition dates. There are no verified control lake-years, so false-alert rates cannot "
               "be estimated, and no operating point was calibrated, so detection timing and warning value are not estimable. All "
               "checkpoints were fitted on years that are also used as comparison references, so out-of-sample seasonal percentiles "
               "would require rolling-origin models, which were not fitted. The 2022 record holds two scenes per site, and 2023 was "
               "sampled more densely than earlier years. The 2017 inventory outline is not a 2018–2023 shoreline and has its own "
               "mapping uncertainty. Snow, ice and cloud are treated as missing rather than modelled. "
               f"{self.n('training.cells[family=primary].stopping.MAX_EPOCH_BUDGET_REACHED', 'int')} of the {self.n('training.cells[family=primary].completed', 'int')} "
               "primary fits stopped at the epoch cap and improved with a longer budget. The positive control is a simple replacement of lake pixels and is easier than real turbid or partly "
               "frozen drainage. The analysis is exploratory and retrospective, and no independent human scientific or statistical "
               "review has taken place.")
        self.H("Conclusions")
        self.P("A small masked spatiotemporal transformer, trained correctly on an incomplete two-lake Sentinel-2 record, uses its "
               "context and reconstructs lake patches at least as well as strong seasonal references, and its reconstruction error "
               "registers a simulated drainage. It did not provide a specific signal of the 2023 South Lhonak outburst: comparable "
               "scores occurred in pre-event monsoon scenes, and a cumulative-sum rule alarmed before the event. Establishing warning "
               "value requires verified control lakes, more events, rolling-origin evaluation and a calibrated operating point. The "
               "source-bound corpus, code, checkpoints and every score are openly available for that work.")

        # ------------------------------------------------ back matter
        self.H("Acknowledgments")
        self.P(NOTICE + " Sentinel-2 Level-2A products were accessed through the Earth Search STAC API (Element 84) on the Registry "
               "of Open Data on AWS. Lake outlines are from the Hi-MAG database (Chen et al.; CC BY 4.0). The use of AI tools is described "
               "in Materials and methods.")
        self.H("References")
        for i, key in enumerate(self.order, start=1):
            self.m.para(f"{i}. {self.refs[key]['vancouver']}")
        unused = sorted(set(self.refs) - set(self.order))
        self.H("Supporting information")
        self.P("S1 File. Analysis protocol. Parent protocol, amendment PLOS-001 with its freeze record, the list of post-freeze "
               "reporting changes and the pre-fix regression-test log (ZIP of JSON and text files).")
        self.P("S1 Table. All training fits. Configuration, seed, selected VAL error, best epoch, epochs completed, stopping status, "
               "validation entries and run time for all fits (CSV).")
        self.P("S2 Table. Scene-level scores. Seed-median Score A, B, C, zone contrast, comparators, CUSUM statistic, candidate water area "
               "and support counts for all scenes (CSV).")
        self.P("S3 Table. Positive-control rows. Clean and perturbed values per seed, scene and replacement fraction, with the "
               "perturbation record (CSV).")
        return unused

    # ---------------------------------------------------------------- figures and tables
    def fig_caption(self, k, title, legend):
        self.m.para(legend, bold_prefix=f"Fig {k}. {title} ")

    def table1(self):
        rows = []
        for s, lab in ((SL, "South Lhonak"), (KT, "Khangchung Tsho")):
            row = [lab]
            for role in ("FIT", "VAL", "EVAL"):
                row.append(f"{self.n(f'corpus.by_site_role.{s}.{role}', 'int')} ({self.n(f'corpus.clear_fraction_median.{s}.{role}', '.2f')})")
            zero = self.packet["corpus"]["zero_observed_patch_scenes"][s]
            row.append(", ".join(f"{self.n(f'corpus.zero_observed_patch_scenes.{s}.{role}', 'int')} {role}" for role in zero))
            rows.append(row)
        self.m.table("Table 1.", "Scenes by site and split (median clear-pixel fraction) and scenes without any observed patch.",
                     ["Site", "FIT", "VAL", "EVAL", "No observed patch"], rows)

    def table2(self):
        rows = []
        order = {"primary": 0, "extended_budget": 1, "capacity": 2, "factorial": 3}
        for cell in sorted(self.packet["training"]["cells"], key=lambda c_: (order[c_["family"]], c_["channels"], c_["zones"], c_["context"], c_["width"])):
            fam, name = cell["family"], cell["cell"]
            sel = f"training.cells[family={fam},cell={name}]"
            label = {"primary": "Primary", "capacity": "Capacity", "factorial": "Factorial", "extended_budget": "Primary, 450 epochs"}[fam]
            cfg = f"{cell['channels']} ch, {'inventory' if cell['zones'] else 'constant'} zones, context {cell['context']}, d = {cell['width']}"
            capn = cell["stopping"].get("MAX_EPOCH_BUDGET_REACHED", 0)
            rows.append([label, cfg, self.n(sel + ".parameter_count", "int"), self.n(sel + ".completed", "int"),
                         f"{self.n(sel + '.mean_val_mse', '.3f')} ({self.n(sel + '.sample_sd_val_mse', '.3f')})",
                         self.n(f"training.validation_reference_mse_by_channels.{cell['channels']}.patch_seasonal_mean", ".3f"),
                         self.n(sel + ".stopping.MAX_EPOCH_BUDGET_REACHED", "int") if capn else "0"])
            for v in (str(cell["channels"]), str(cell["context"]), str(cell["width"])):
                self.fixed.add(v)
        self.m.table("Table 2.", "Training cells: selected VAL error, mean (sample SD) over seeds, beside the per-patch seasonal FIT mean "
                     "on the same VAL entries.", ["Family", "Configuration", "Parameters", "Seeds", "VAL MSE", "Seasonal reference",
                                                  "Fits at epoch cap"], rows)

    def table3(self):
        names = [("model", "T-MAE (cross-mask)"), ("model_visible_inputs_zeroed", "T-MAE, visible inputs set to 0"),
                 ("model_all_context_hidden", "T-MAE, all context hidden"),
                 ("patch_seasonal_mean_plus_visible_offset", "Seasonal mean + visible offset"),
                 ("patch_seasonal_mean", "Per-patch seasonal FIT mean"), ("patch_mean", "Per-patch FIT mean"),
                 ("calendar_site_mean", "Calendar-only site mean"), ("fit_median", "FIT median (zero)")]
        rows = []
        for key, lab in names:
            row = [lab] + [self.rc(role, key) for role in ("FIT", "VAL", "EVAL")]
            for role in ("VAL", "EVAL"):
                b_ = f"reconstruction.differences.{role}.{key}"
                row.append("—" if key == "model" else
                           f"{self.n(b_ + '.estimate', 'signed3')} [{self.n(b_ + '.lower', 'signed3')}, {self.n(b_ + '.upper', 'signed3')}]")
            rows.append(row)
        self.m.table("Table 3.", "Pooled mean squared error on identical hidden entries, and model minus comparator with 95% "
                     "moving-block bootstrap intervals in brackets.", ["Predictor", "FIT", "VAL", "EVAL", "Difference, VAL", "Difference, EVAL"], rows)

    def table4(self):
        rows = []
        groups = [("AUTUMN_FIT_2018_2021", "autumn FIT"), ("VAL", "VAL"), ("EVAL", "EVAL")]
        for row_def in self.packet["scores"]["table"]:
            fld = row_def["field"]
            row = [row_def["label"]]
            for s in (SL, KT):
                for g, _ in groups:
                    base = f"scores.table[field={fld}].{s}|{g}"
                    if self.packet["scores"]["table"][[x["field"] for x in self.packet["scores"]["table"]].index(fld)][f"{s}|{g}"]["n"] == 0:
                        row.append("not estimable")
                        continue
                    row.append(f"{self.n(base + '.median', '.3f')} [{self.n(base + '.q1', '.3f')}, {self.n(base + '.q3', '.3f')}] ({self.n(base + '.n', 'int')})")
            rows.append(row)
        hdr = ["Quantity"] + [f"{'SL' if s == SL else 'KT'} {lab}" for s in (SL, KT) for _, lab in groups]
        self.m.table("Table 4.", "Scene-level medians [interquartile range] (number of scenes) after taking the median over ten seeds. "
                     "SL, South Lhonak; KT, Khangchung Tsho; autumn FIT, October–November 2018–2021 (in-sample for the model); VAL, "
                     "held-out pre-event scenes; EVAL, post-event scenes.", hdr, rows)

    def table5(self):
        rows = []
        for s, lab in ((SL, "South Lhonak"), (KT, "Khangchung Tsho")):
            for role in ("FIT", "VAL", "EVAL"):
                a = f"cusum.alarms.{s}|{role}"
                first = self.packet["cusum"]["alarms"][f"{s}|{role}"]["first_majority"]
                rows.append([lab, role, self.n(a + ".scenes", "int"), self.n(a + ".scenes_any_seed", "int"),
                             self.n(a + ".scenes_majority", "int"), self.fx(first, "packet cusum") if first else "—"])
        self.m.table("Table 5.", f"CUSUM alarm declarations on Score C (κ = 0.5, h = 4, no resets): scenes in which the statistic "
                     f"reached h for at least one seed and for a majority ({self.n('cusum.majority_seeds', 'int')} or more of 10) of seeds.",
                     ["Site", "Split", "Scenes", "Alarm, ≥1 seed", "Alarm, majority", "First majority alarm"], rows)

    def table6(self):
        labels = {"score_a_zone0_1": "Score A", "score_b_zone0_1": "Score B", "score_c": "Score C",
                  "water_candidate_area_km2": "Candidate water area", "seasonal_climatology_residual": "Seasonal residual",
                  "spatial_persistence_residual": "Temporal persistence", "isolation_forest_score": "Isolation Forest",
                  "one_class_svm_score": "One-class SVM", "missingness_only_score": "Missingness only"}
        rows = []
        for meth, lab in labels.items():
            row = [lab]
            for frac in ("0.1", "0.5", "1.0"):
                row.append(f"{self.pc(meth, frac, 'clean_detection_rate')} → {self.pc(meth, frac, 'perturbed_detection_rate')}")
            row.append(self.pc(meth, "1.0", "median_paired_change", "signed3"))
            rows.append(row)
        self.m.table("Table 6.", "Positive control: share of scenes beyond the FIT threshold, unperturbed → perturbed, by fraction of "
                     "lake pixels replaced, and the median paired change at full replacement.",
                     ["Method", "10%", "50%", "100%", "Median change, 100%"], rows)

    def table7(self):
        rows = []
        for key in ("E1", "E2", "E3", "E4", "E5"):
            e = self.packet["estimands"][key]
            rows.append([e["name"], e["status"].replace("_", " ").lower(), e["reason"]])
        self.m.table("Table 7.", "Status of the protocol's estimands.", ["Estimand", "Status", "Reason"], rows)


def write_si(packet):
    SUB.mkdir(parents=True, exist_ok=True)
    run = json.loads((OUT_ANALYSIS / "training/run_manifest.json").read_text())
    with (SUB / "S1_Table.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "cell", "channels", "zones", "context", "width", "seed", "max_epochs", "status", "best_val_mse",
                    "best_epoch", "epochs_completed", "stopping_status", "validation_blocks", "validation_target_entries", "elapsed_seconds"])
        for j in run["jobs"]:
            w.writerow([j["family"], j["cell"], j["channels"], j["zones"], j["context"], j["width"], j["seed"], j["max_epochs"],
                        j["status"], j.get("best_val_loss"), j.get("best_epoch"), j.get("epochs_completed"), j.get("stopping_status"),
                        j.get("validation_block_count"), j.get("validation_target_entries"), j.get("elapsed_seconds")])
    preds = json.loads((OUT_ANALYSIS / "scores/predictions.json").read_text())
    import numpy as np
    fields = ["score_a_zone0_1", "score_a_zone2", "score_a_zone_contrast", "score_b_zone0_1", "score_c", "page_cusum.statistic",
              "water_candidate_area_km2", "baselines.seasonal_climatology_residual", "baselines.spatial_persistence_residual",
              "baselines.isolation_forest_score", "baselines.one_class_svm_score", "baselines.missingness_only_score"]
    by = {}
    for r_ in preds:
        by.setdefault(r_["observation_id"], []).append(r_)
    with (SUB / "S2_Table.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["observation_id", "site_id", "acquired_at_utc", "split", "observed_patches", "observed_zone01_patches",
                    "score_a_status", "score_b_status"] + [f"{x}_seed_median" for x in fields] + [f"cusum_alarm_seeds_of_{len(set(r_['seed'] for r_ in preds))}"])
        for oid, rows in sorted(by.items(), key=lambda kv: (kv[1][0]["site_id"], kv[1][0]["acquired_at"])):
            first = rows[0]
            vals = []
            for fld in fields:
                xs = []
                for r_ in rows:
                    v = r_
                    for k in fld.split("."):
                        v = v.get(k) if isinstance(v, dict) else None
                    if v is not None:
                        xs.append(v)
                vals.append(float(np.median(xs)) if xs else "")
            w.writerow([oid, first["site_id"], first["acquired_at"], first["role"], first["observed_patches"],
                        first["observed_zone01_patches"], first["score_a_status"], first["score_b_status"], *vals,
                        sum(r_["page_cusum"]["alarm_declared"] for r_ in rows)])
    pc = json.loads((OUT_ANALYSIS / "scores/positive_control.json").read_text())
    keys = ["score_a_zone0_1", "score_b_zone0_1", "score_c", "water_candidate_area_km2", "seasonal_climatology_residual",
            "spatial_persistence_residual", "isolation_forest_score", "one_class_svm_score", "missingness_only_score"]
    with (SUB / "S3_Table.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["origin", "seed", "observation_id", "site_id", "split", "fraction", "status", "replaced_clear_pixels"] +
                   [f"clean_{k}" for k in keys] + [f"perturbed_{k}" for k in keys])
        for r_ in pc:
            w.writerow(["constructed_positive_control", r_["seed"], r_["observation_id"], r_["site_id"], r_["role"], r_["fraction"],
                        r_["status"], r_["injection"].get("replaced_clear_pixels", "")] +
                       [r_.get("clean", {}).get(k, "") for k in keys] + [r_.get("perturbed", {}).get(k, "") for k in keys])
    import zipfile
    with zipfile.ZipFile(SUB / "S1_File.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT / "docs/protocol").iterdir()):
            z.write(p, f"protocol/{p.name}")


def write_cover_letter():
    from docx import Document
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    paras = [
        "Dear Editor,",
        f"Please consider the enclosed manuscript, \u201c{TITLE}\u201d, for publication in PLOS ONE as a Research Article.",
        "The study asks whether a small masked spatiotemporal transformer, trained on an incomplete Sentinel-2 record of two "
        "Sikkim glacial lakes, adds information beyond parameter-free seasonal references, and what its scores showed around the "
        "October 2023 South Lhonak outburst. We report the results whichever way they fell: the model uses its context and "
        "registers a constructed drainage, but on held-out pre-event scenes it is not distinguishable from a seasonal reference "
        "that uses the same visible context, and its anomaly scores did not give a specific signal of the outburst. Four of the "
        "five quantities we set out to estimate are reported as not estimable, with reasons. We believe this negative and "
        "bounded result is useful to the growing literature applying learned models to glacial-lake hazards, and that PLOS ONE's "
        "focus on rigour rather than perceived impact makes it the right venue.",
        "This analysis was originally planned for the ICT-CEEL 2027 conference, but it was not submitted there or to any "
        "other venue. An earlier version of the analysis contained a training defect, corrected here, and none of its results "
        "are used. An earlier, superseded snapshot of the code was public in the author's GitHub repository; the repository now "
        "carries the release described in the manuscript.",
        "All data, code, model checkpoints and score tables are openly available (see the Data Availability Statement). The "
        "analysis protocol amendment was frozen, with code digests, before the models were trained, and every number in the "
        "manuscript is regenerated and checked from the retained prediction rows by a verification script. AI tools used in "
        "the work, and the author's own contribution, are described in the Methods.",
        "The manuscript has not been published and is not under consideration elsewhere. The work received no specific "
        "funding, and the author has no competing interests. As an author based at a Nepali institution without external "
        "funding for this study, I am requesting publication-fee assistance under PLOS's Research4Life Group A policy.",
        "Thank you for your consideration.",
        "Sincerely,",
        "Aditya Shrestha",
        "Department of Computer Science and Engineering, School of Engineering, Kathmandu University, Dhulikhel, Nepal",
        "as03233622@student.ku.edu.np · ORCID 0009-0009-2913-4823",
    ]
    for text in paras:
        doc.add_paragraph(text)
    doc.save(SUB / "Cover_Letter.docx")


def main() -> int:
    b = Builder()
    unused = b.build()
    SUB.mkdir(parents=True, exist_ok=True)
    out = SUB / "Manuscript.docx"
    b.m.doc.core_properties.title = TITLE
    b.m.doc.core_properties.author = "Aditya Shrestha"
    b.m.save(out)
    write_si(b.packet)
    write_cover_letter()
    for k in range(1, 6):
        shutil.copyfile(FIGS / f"Fig{k}.tif", SUB / f"Fig{k}.tif")
    ledger = {"packet_sha256": sha(PACKET), "manuscript_sha256": sha(out), "entries": b.L.entries,
              "fixed_tokens": sorted(b.fixed), "reference_order": b.order, "unused_verified_references": unused}
    (SUB / "number_ledger.json").write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"manuscript": out.relative_to(ROOT).as_posix(), "references": len(b.order), "unused": unused,
                      "ledger_entries": len(b.L.entries)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
