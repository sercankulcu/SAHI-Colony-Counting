#!/usr/bin/env python3
"""Run 5-fold CV on sliced YOLO tiles while grouping by original image ID.

This avoids tile-level leakage: all tiles derived from the same original Petri dish
image are kept in the same fold.
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path

import yaml
from sklearn.model_selection import KFold
from ultralytics import YOLO

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
CLASS_NAMES = [
    'Actinobacillus_equuli', 'Actinobacillus_pleuropneumoniae', 'Aeromonas_hydrophila',
    'Bacillus_cereus', 'Bibersteinia_trehalosi', 'Bordetella_bronchiseptica',
    'Brucella_ovis', 'Clostridium_perfringens', 'Corynebacterium_pseudotuberculosis',
    'Erysipelothrix_rhusiopathiae', 'Escherichia_coli', 'Glaesserella_parasuis',
    'Klebsiella_pneumoniae', 'Listeria_monocytogenes', 'Paenibacillus_larvae',
    'Pasteurella_multocida', 'Proteus_mirabilis', 'Pseudomonas_aeruginosa',
    'Rhodococcus_equi', 'Salmonella_enterica', 'Staphylococcus_aureus',
    'Staphylococcus_hyicus', 'Streptococcus_agalactiae', 'Trueperella_pyogenes'
]


def original_id_from_tile(stem: str) -> str:
    # Supports names like original_slice_0001, original_tile_12, original_sp01, etc.
    return re.sub(r'(_slice_\d+|_tile_\d+)$', '', stem)


def list_tile_groups(images_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() in IMAGE_EXTS:
            groups[original_id_from_tile(p.stem)].append(p)
    if not groups:
        raise FileNotFoundError(f'No tile images found in {images_dir}')
    return dict(groups)


def copy_tiles(image_files: list[Path], labels_dir: Path, fold_dir: Path, split: str) -> None:
    for image_path in image_files:
        label_path = labels_dir / f'{image_path.stem}.txt'
        target_img = fold_dir / 'images' / split / image_path.name
        target_lbl = fold_dir / 'labels' / split / f'{image_path.stem}.txt'
        target_img.parent.mkdir(parents=True, exist_ok=True)
        target_lbl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_img)
        if label_path.exists():
            shutil.copy2(label_path, target_lbl)
        else:
            target_lbl.write_text('', encoding='utf-8')


def create_yaml(fold_dir: Path) -> Path:
    data = {
        'path': str(fold_dir.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(CLASS_NAMES),
        'names': {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    yaml_path = fold_dir / 'data.yaml'
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images-dir', type=Path, required=True, help='Folder containing all sliced tile images.')
    parser.add_argument('--labels-dir', type=Path, required=True, help='Folder containing matching YOLO labels.')
    parser.add_argument('--output-root', type=Path, default=Path('cross_validation'))
    parser.add_argument('--model', default='yolo11n.pt')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='0')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    groups = list_tile_groups(args.images_dir)
    original_ids = sorted(groups.keys())
    kfold = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(original_ids), start=1):
        fold_dir = args.output_root / f'fold_{fold_idx}'
        if fold_dir.exists():
            shutil.rmtree(fold_dir)
        train_ids = [original_ids[i] for i in train_idx]
        val_ids = [original_ids[i] for i in val_idx]
        train_files = [p for oid in train_ids for p in groups[oid]]
        val_files = [p for oid in val_ids for p in groups[oid]]
        copy_tiles(train_files, args.labels_dir, fold_dir, 'train')
        copy_tiles(val_files, args.labels_dir, fold_dir, 'val')
        data_yaml = create_yaml(fold_dir)

        print(f'Fold {fold_idx}: {len(train_ids)} train original images / {len(val_ids)} val original images')
        print(f'Fold {fold_idx}: {len(train_files)} train tiles / {len(val_files)} val tiles')
        model = YOLO(args.model)
        model.train(
            data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            device=args.device, project=str(args.output_root / 'training_runs'),
            name=f'fold_{fold_idx}', exist_ok=True,
        )
        summary_rows.append({
            'fold': fold_idx,
            'train_original_images': len(train_ids),
            'val_original_images': len(val_ids),
            'train_tiles': len(train_files),
            'val_tiles': len(val_files),
            'data_yaml': str(data_yaml),
        })

    summary_path = args.output_root / 'fold_summary.csv'
    with summary_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f'Saved: {summary_path}')


if __name__ == '__main__':
    main()
