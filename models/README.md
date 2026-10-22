# Model files

FusionGuard uses ArcFace identity embeddings and a separate Mahalanobis geometry model. Personal enrolment templates and captured face images are created at runtime and are not packaged with the repository.

## Included Mahalanobis model

The repository includes [`mahalanobis_model_full_500.npz`](../mahalanobis_model_full_500.npz) in the project root. The archive is 1,538 bytes and contains three numerical arrays:

| Array | Shape | Role in the comparison code |
| --- | --- | --- |
| `mean` | `(9,)` | Centre for the nine-dimensional normalised geometry vector |
| `std` | `(9,)` | Per-dimension scaling values |
| `inv_cov` | `(9, 9)` | Inverse covariance matrix used for the Mahalanobis distance |

[`fusionguard_multimodal.py`](../fusionguard_multimodal.py) loads the archive with NumPy, standardises the geometry vectors with `mean` and `std`, and calculates their distance using `inv_cov`. Zero standard-deviation entries are replaced with `1.0` during standardisation. The file contains model parameters and has no identity-label or enrolled-face embedding arrays.

The default path is defined in [`config.py`](../config.py). `MAHALANOBIS_MODEL_PATH` can select a different archive with the same required arrays and compatible dimensions.

### Structural checks

From the project root:

```sh
python tools/check_model.py
```

The utility reports the required array keys and their shapes. The recorded structural validation also checked the expected dimensions and confirmed that all array values are finite.

These checks do not establish training provenance, dataset composition, sample count, model calibration or recognition accuracy. No training-data claim is inferred from the filename.

If the configured file is absent, the recognition code uses an identity-covariance fallback: the Euclidean norm of the difference between the normalised geometry vectors. That fallback changes the distance calculation and is reported in comparison results as `identity-covariance fallback`.

## ArcFace weights

ArcFace is loaded through DeepFace. Its network weights are stored in DeepFace's model cache separately from this repository and the Mahalanobis archive.

With the full application dependencies installed, model preparation runs from the project root:

```sh
python prepare_arcface.py
```

The script loads the ArcFace recognition model. The first run may download the required weights and therefore requires network access when the weights are not already cached. The complete runtime setup is described in [Local deployment](../docs/LOCAL_DEPLOYMENT.md).

## Runtime enrolment data

Four-view enrolment produces per-user feature templates containing ArcFace embeddings, geometry values and LBP histograms. The application stores those compressed templates in MySQL and the associated images under `static/uploads/enrollment/`. Live recognition and incident images use the corresponding `live/` and `incidents/` directories.

These runtime records are separate from the included model parameters. No personal face templates, capture images or populated database are included in the repository.

## Related documentation

- [Project overview](../README.md)
- [Architecture](../docs/ARCHITECTURE.md)
- [Testing](../docs/TESTING.md)
