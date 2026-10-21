"""COCO bbox metrics with an explicit, reproducible area protocol.

All six AP values are extracted from the same COCOeval precision tensor.  They
are fractions (0..1), not percentages; an undefined AP is returned as None.
Areas use original-image GT bbox width * height, never network-input pixels.
No ground truth or predictions are prefiltered by object area: COCOeval owns
matching, ignored-GT handling, and out-of-range unmatched-detection handling.

Only standard-library modules are imported at module load time.  Runtime and
synthetic checks require NumPy and pycocotools on the evaluation server.
"""

import copy
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path


METRIC_KEYS = ('map50_95', 'map50', 'map75', 'aps', 'apm', 'apl')
AREA_MODES = ('custom', 'standard')


def _area_protocol(area_mode):
    if area_mode not in AREA_MODES:
        raise ValueError('area_mode must be custom or standard, got %r' % area_mode)
    limits = (64, 192) if area_mode == 'custom' else (32, 96)
    small, large = (float(value ** 2) for value in limits)
    return limits, [[0.0, 1e10], [0.0, small],
                    [small, large], [large, 1e10]]


def configure_evaluator(coco_gt, coco_dt, area_mode='custom'):
    """Return COCOeval with every relevant bbox-evaluation parameter explicit.

    This deliberately overrides the installed package's defaults, including
    any earlier manual modifications to pycocotools' area thresholds.
    """
    import numpy as np
    from pycocotools.cocoeval import COCOeval

    _, ranges = _area_protocol(area_mode)
    evaluator = COCOeval(coco_gt, coco_dt, iouType='bbox')
    params = evaluator.params
    params.imgIds = sorted(int(value) for value in coco_gt.getImgIds())
    params.catIds = sorted(int(value) for value in coco_gt.getCatIds())
    params.iouType = 'bbox'
    params.useSegm = None
    params.useCats = 1
    params.iouThrs = np.linspace(0.50, 0.95, 10, endpoint=True)
    params.recThrs = np.linspace(0.0, 1.0, 101, endpoint=True)
    params.maxDets = [1, 10, 100]
    params.areaRng = ranges
    params.areaRngLbl = ['all', 'small', 'medium', 'large']
    return evaluator


def _integer(value, label, positive=False):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError('%s must be an integer, got %r' % (label, value))
    if positive and value <= 0:
        raise ValueError('%s must be positive' % label)
    return value


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('%s must be a finite numeric value' % label)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('%s must be finite' % label)
    return result


def _bbox(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError('%s must be [x, y, width, height]' % label)
    result = [_finite(item, label) for item in value]
    if result[2] <= 0 or result[3] <= 0:
        raise ValueError('%s width and height must be positive' % label)
    return result


def _validated_inputs(gt_dataset_dict, predictions):
    """Copy and validate input without changing original-image coordinates."""
    if not isinstance(gt_dataset_dict, dict) or not isinstance(predictions, list):
        raise ValueError('Expected a COCO dataset dict and a prediction list')
    dataset = copy.deepcopy(gt_dataset_dict)
    for key in ('images', 'categories', 'annotations'):
        if not isinstance(dataset.get(key), list):
            raise ValueError('Ground truth %s must be a list' % key)
    if not dataset['images'] or not dataset['categories']:
        raise ValueError('Ground truth must contain images and categories')
    dataset.setdefault('info', {})
    dataset.setdefault('licenses', [])

    image_ids, category_ids, annotation_ids = set(), set(), set()
    for item in dataset['images']:
        image_id = _integer(item['id'], 'image id')
        if image_id in image_ids:
            raise ValueError('Duplicate image id: %d' % image_id)
        image_ids.add(image_id)
        for dimension in ('width', 'height'):
            if _finite(item[dimension], 'image ' + dimension) <= 0:
                raise ValueError('Image dimensions must be positive')
    for item in dataset['categories']:
        category_id = _integer(item['id'], 'category id')
        if category_id in category_ids:
            raise ValueError('Duplicate category id: %d' % category_id)
        category_ids.add(category_id)
        if not isinstance(item.get('name'), str):
            raise ValueError('Each category must have a string name')
    for item in dataset['annotations']:
        annotation_id = _integer(item['id'], 'annotation id', positive=True)
        if annotation_id in annotation_ids:
            raise ValueError('Duplicate annotation id: %d' % annotation_id)
        annotation_ids.add(annotation_id)
        if _integer(item['image_id'], 'annotation image_id') not in image_ids:
            raise ValueError('Annotation references an unknown image')
        if _integer(item['category_id'], 'annotation category_id') not in category_ids:
            raise ValueError('Annotation references an unknown category')
        item['bbox'] = _bbox(item['bbox'], 'GT bbox')
        bbox_area = item['bbox'][2] * item['bbox'][3]
        if not math.isfinite(bbox_area):
            raise ValueError('GT bbox area is not finite')
        area = _finite(item.get('area', bbox_area), 'GT area')
        if not math.isclose(area, bbox_area, rel_tol=1e-8, abs_tol=1e-7):
            raise ValueError('GT area must equal original-image bbox width * height')
        item['area'] = bbox_area
        crowd = item.get('iscrowd', 0)
        if crowd not in (0, 1, False, True):
            raise ValueError('iscrowd must be 0 or 1')
        item['iscrowd'] = int(crowd)
        if item.get('ignore', 0) and not crowd:
            raise ValueError('Explicit non-crowd ignore annotations are not supported '
                             'by this EPFD bbox protocol')

    cleaned_predictions = []
    for item in predictions:
        image_id = _integer(item['image_id'], 'prediction image_id')
        category_id = _integer(item['category_id'], 'prediction category_id')
        if image_id not in image_ids or category_id not in category_ids:
            raise ValueError('Prediction references an unknown image or category')
        bbox = _bbox(item['bbox'], 'prediction bbox')
        score = _finite(item['score'], 'prediction score')
        if not 0.0 <= score <= 1.0:
            raise ValueError('Prediction confidence must lie in [0, 1]')
        cleaned_predictions.append({'image_id': image_id,
                                    'category_id': category_id,
                                    'bbox': bbox, 'score': score})
    return dataset, cleaned_predictions


def _metric_values(evaluator, class_index=None):
    import numpy as np

    params = evaluator.params
    precision = evaluator.eval['precision']
    max_index = params.maxDets.index(100)
    expected = (10, 101, len(params.catIds), 4, 3)
    if tuple(precision.shape) != expected:
        raise RuntimeError('Unexpected COCO precision tensor shape: %r' %
                           (tuple(precision.shape),))

    def ap(area='all', iou=None):
        values = precision[:, :, :, params.areaRngLbl.index(area), max_index]
        if class_index is not None:
            values = values[:, :, class_index:class_index + 1]
        if iou is not None:
            indices = np.flatnonzero(np.isclose(params.iouThrs, iou,
                                                rtol=0.0, atol=1e-12))
            if len(indices) != 1:
                raise RuntimeError('Required IoU threshold is missing or duplicated')
            values = values[indices, ...]
        valid = values[values > -1]
        if not valid.size:
            return None
        result = float(np.mean(valid))
        if not math.isfinite(result) or not -1e-12 <= result <= 1.0 + 1e-12:
            raise RuntimeError('COCOeval returned an invalid AP: %r' % result)
        return result

    return {'map50_95': ap(), 'map50': ap(iou=0.50), 'map75': ap(iou=0.75),
            'aps': ap('small'), 'apm': ap('medium'), 'apl': ap('large')}


def _source_info(evaluator):
    module_path = inspect.getsourcefile(type(evaluator))
    source_hash = None
    if module_path and Path(module_path).is_file():
        source_hash = hashlib.sha256(Path(module_path).read_bytes()).hexdigest()
    try:
        version = importlib.metadata.version('pycocotools')
    except importlib.metadata.PackageNotFoundError:
        version = 'unknown'
    return {'package': 'pycocotools', 'version': version,
            'cocoeval_source': Path(module_path).name if module_path else None,
            'cocoeval_sha256': source_hash}


def evaluate_coco(gt_dataset_dict, predictions, area_mode='custom'):
    """Evaluate six AP metrics and per-class metrics; return JSON-safe data.

    ``area_counts`` counts non-crowd GT annotations.  COCO's area intervals
    include BOTH endpoints, so a GT exactly at a boundary appears in each
    adjacent area group.  This is intentional and matches COCOeval, not a
    disjoint dataset-size partition.  No valid GT in a group yields None AP;
    valid GT but no detections yields 0 AP.
    """
    from pycocotools.coco import COCO

    limits, _ = _area_protocol(area_mode)
    dataset, detections = _validated_inputs(gt_dataset_dict, predictions)
    coco_gt = COCO()
    coco_gt.dataset = dataset
    coco_gt.createIndex()
    if detections:
        # COCO.loadRes mutates its list argument by adding areas and ids.
        coco_dt = coco_gt.loadRes(copy.deepcopy(detections))
    else:
        # Official loadRes([]) indexes anns[0]; construct an empty result index.
        coco_dt = COCO()
        coco_dt.dataset = {'info': copy.deepcopy(dataset['info']),
                           'images': copy.deepcopy(dataset['images']),
                           'categories': copy.deepcopy(dataset['categories']),
                           'annotations': []}
        coco_dt.createIndex()
    evaluator = configure_evaluator(coco_gt, coco_dt, area_mode)
    evaluator.evaluate()
    evaluator.accumulate()
    # Do not depend on summarize()/stats or installed default area thresholds.
    metrics = _metric_values(evaluator)
    params = evaluator.params
    categories = {item['id']: item['name'] for item in dataset['categories']}
    per_class = []
    for index, class_id in enumerate(params.catIds):
        row = {'class_id': int(class_id), 'name': categories[int(class_id)]}
        row.update(_metric_values(evaluator, index))
        per_class.append(row)
    area_counts = {}
    for label, (lower, upper) in zip(params.areaRngLbl, params.areaRng):
        area_counts[label] = sum(1 for item in dataset['annotations']
                                 if not item['iscrowd'] and
                                 lower <= item['area'] <= upper)
    parameter_report = {
        'iouType': 'bbox', 'useCats': 1,
        'image_ids': [int(value) for value in params.imgIds],
        'category_ids': [int(value) for value in params.catIds],
        'iou_thresholds': [float(value) for value in params.iouThrs],
        'recall_thresholds': [float(value) for value in params.recThrs],
        'max_detections': list(params.maxDets), 'reported_max_detections': 100,
        'area_mode': area_mode, 'linear_area_boundaries': list(limits),
        'area_ranges_px2': copy.deepcopy(params.areaRng),
        'area_labels': list(params.areaRngLbl),
        'area_coordinate_system': 'original image, before detector resizing',
        'ground_truth_area': 'bbox width * bbox height in original-image pixels',
        'area_boundary_rule': 'inclusive endpoints; exact-boundary GT may belong '
                              'to both adjacent groups (COCOeval semantics)',
        'area_count_definition': 'non-crowd GT eligible by area; not detection count',
        'prediction_area_prefiltering': False,
        'undefined_metric': None,
        'metric_unit': 'fraction; multiply by 100 for percent',
        'scale_metric_names': ['AP_S*', 'AP_M*', 'AP_L*'] if area_mode == 'custom'
                              else ['AP_S', 'AP_M', 'AP_L'],
        'installed_defaults_overridden': True,
        'implementation': _source_info(evaluator),
    }
    result = {'metrics': metrics, 'per_class': per_class,
              'area_counts': area_counts, 'eval_parameters': parameter_report,
              'evaluated_images': len(params.imgIds), 'detections': len(detections),
              'ground_truth_annotations': len(dataset['annotations'])}
    json.dumps(result, allow_nan=False)
    return result


def self_check():
    """Run synthetic COCO checks on the server; no checkpoint or file writes."""
    from unittest.mock import patch

    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import Params

    def require(condition, message):
        if not condition:
            raise AssertionError(message)

    def fixture(sizes, absent_category=False):
        ground_truth = {'info': {}, 'images': [], 'annotations': [],
                        'categories': [{'id': 7, 'name': 'fitting'}]}
        if absent_category:
            ground_truth['categories'].append({'id': 31, 'name': 'absent'})
        predictions = []
        for index, size in enumerate(sizes, 1):
            ground_truth['images'].append({'id': index, 'width': 1024, 'height': 1024})
            box = [20.0, 20.0, float(size), float(size)]
            ground_truth['annotations'].append(
                {'id': index, 'image_id': index, 'category_id': 7,
                 'bbox': box, 'area': float(size * size), 'iscrowd': 0})
            predictions.append({'image_id': index, 'category_id': 7,
                                'bbox': list(box), 'score': 0.9})
        return ground_truth, predictions

    gt, predictions = fixture([16, 80, 240])
    original = copy.deepcopy((gt, predictions))
    for mode in AREA_MODES:
        result = evaluate_coco(gt, predictions, mode)
        require(all(abs(result['metrics'][key] - 1.0) < 1e-10
                    for key in METRIC_KEYS), 'Perfect detections must have AP=1')
        require(result['area_counts'] == {'all': 3, 'small': 1, 'medium': 1, 'large': 1},
                'Three-scale counts incorrect for ' + mode)
        empty = evaluate_coco(gt, [], mode)
        require(all(empty['metrics'][key] == 0.0 for key in METRIC_KEYS),
                'No detections with valid GT must have AP=0')
    require((gt, predictions) == original, 'Evaluation mutated caller inputs')

    single_gt, single_predictions = fixture([48], absent_category=True)
    custom = evaluate_coco(single_gt, single_predictions, 'custom')
    standard = evaluate_coco(single_gt, single_predictions, 'standard')
    require(custom['area_counts']['small'] == 1 and
            standard['area_counts']['medium'] == 1, 'Custom area switch failed')
    require(custom['metrics']['apm'] is None and custom['metrics']['apl'] is None
            and standard['metrics']['aps'] is None, 'Empty area AP must be None')
    absent = next(row for row in custom['per_class'] if row['class_id'] == 31)
    require(all(absent[key] is None for key in METRIC_KEYS),
            'Absent category AP must be None, not zero')
    boundary_gt, boundary_predictions = fixture([64, 192])
    boundary = evaluate_coco(boundary_gt, boundary_predictions, 'custom')
    require(boundary['area_counts'] == {'all': 2, 'small': 1, 'medium': 2, 'large': 1},
            'COCO inclusive boundary counts incorrect')

    # A high-scoring false positive of large area must not damage AP_S, but it
    # must remain present in all-area AP.  This tests evaluator ignore semantics.
    ignore_gt, ignore_predictions = fixture([16])
    ignore_predictions.insert(0, {'image_id': 1, 'category_id': 7,
                                  'bbox': [600, 600, 240, 240], 'score': 0.99})
    ignored = evaluate_coco(ignore_gt, ignore_predictions, 'custom')
    require(abs(ignored['metrics']['aps'] - 1.0) < 1e-10 and
            ignored['metrics']['map50_95'] < 0.6 and ignored['detections'] == 2,
            'Out-of-area detection ignore semantics changed')

    original_set = Params.setDetParams

    def modified_defaults(params):
        original_set(params)
        params.areaRng = [[0, 1], [0, 0.1], [0.1, 0.2], [0.2, 1]]
        params.areaRngLbl = ['wrong'] * 4
        params.iouThrs = np.array([0.4])
        params.recThrs = np.array([0.0, 1.0])
        params.maxDets = [1, 2, 3]
        params.useCats = 0

    indexed_gt = COCO()
    indexed_gt.dataset = copy.deepcopy(gt)
    indexed_gt.createIndex()
    indexed_dt = indexed_gt.loadRes(copy.deepcopy(predictions))
    with patch.object(Params, 'setDetParams', modified_defaults):
        for mode in AREA_MODES:
            evaluator = configure_evaluator(indexed_gt, indexed_dt, mode)
            _, expected_ranges = _area_protocol(mode)
            require(evaluator.params.areaRng == expected_ranges and
                    evaluator.params.areaRngLbl == ['all', 'small', 'medium', 'large']
                    and evaluator.params.maxDets == [1, 10, 100]
                    and evaluator.params.useCats == 1
                    and len(evaluator.params.iouThrs) == 10
                    and len(evaluator.params.recThrs) == 101,
                    'Installed package defaults were not overridden')

    print('PASS: six COCO AP metrics, custom/standard original-coordinate areas, '
          'empty detections, undefined groups, inclusive boundaries, per-class '
          'metrics, ignored out-of-area detections, and explicit default overrides.')
    print('No detector inference, training, or file writes were executed.')
    return True


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true', required=True)
    parser.parse_args()
    self_check()
