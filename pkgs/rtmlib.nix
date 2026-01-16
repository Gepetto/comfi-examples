{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  setuptools,
  wheel,
  numpy,
  onnxruntime,
  opencv-contrib-python,
  opencv-python,
  tqdm,
}:

buildPythonPackage rec {
  pname = "rtmlib";
  version = "0.0.14";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "Tau-J";
    repo = "rtmlib";
    tag = version;
    hash = "sha256-MV2uISbJcgEuKCdFC8lzHO3RLFYalClMnlaTqpxgIiI=";
  };

  postPatch = ''
    substituteInPlace setup.py \
      --replace-fail "locals()['short_version']" "'${version}'" \
      --replace-fail "locals()['__version__']" "'${version}'"
  '';

  build-system = [
    setuptools
    wheel
  ];

  dependencies = [
    numpy
    onnxruntime
    opencv-contrib-python
    opencv-python
    tqdm
  ];

  pythonRelaxDeps = [
    "onnxruntime"
  ];

  pythonImportsCheck = [
    "rtmlib"
  ];

  meta = {
    description = "RTMPose series (RTMPose, DWPose, RTMO, RTMW) without mmcv, mmpose, mmdet etc";
    homepage = "https://github.com/Tau-J/rtmlib";
    license = lib.licenses.asl20;
    maintainers = with lib.maintainers; [ nim65s ];
  };
}
