{
  lib,
  buildPythonPackage,

  # build-system
  uv-build,

  # dependencies
  example-robot-data,
  httpx,
  imageio,
  matplotlib,
  meshcat,
  meshcat-shapes,
  onnxruntime,
  opencv-contrib-python,
  opencv-python,
  pandas,
  pinocchio,
  casadi,
  pyyaml,
  rtmlib,
  scipy,
  tqdm,
}:
buildPythonPackage {
  name = "comfi-examples";
  version = "0.1.0";
  pyproject = true;

  src = lib.fileset.toSource {
    root = ./..;
    fileset = lib.fileset.unions [
      ../comfi_examples
      ../scripts
      ../pyproject.toml
      ../README.md
    ];
  };

  build-system = [
    uv-build
  ];

  dependencies = [
    example-robot-data
    httpx
    imageio
    matplotlib
    meshcat
    meshcat-shapes
    onnxruntime
    opencv-contrib-python
    opencv-python
    pandas
    pinocchio
    casadi
    pyyaml
    rtmlib
    scipy
    tqdm
  ];

  pythonRelaxDeps = [
    "matplotlib"
    "onnxruntime"
    "opencv-python"
    "opencv-contrib-python"
  ];

  pythonRemoveDeps = [
    "casadi"
    "example-robot-data"
    "pin"
  ];

  pythonImportsCheck = [
    "comfi_examples"
  ];

  meta = {
    description = "human-robot Collaboration Oriented Markerless For Industry dataset utils";
    homepage = "https://github.com/gepetto/comfi-examples";
    license = lib.licenses.bsd2;
    maintainers = with lib.maintainers; [ nim65s ];
  };
}
