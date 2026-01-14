{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  flit-core,
  meshcat,
  sphinx,
}:

buildPythonPackage rec {
  pname = "meshcat-shapes";
  version = "1.0.0";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "stephane-caron";
    repo = "meshcat-shapes";
    rev = "v${version}";
    hash = "sha256-4T9uT7WhRCpsS1jyghv6bkAxQ/EkP+8Vgp1vc7/kGCk=";
  };

  build-system = [
    flit-core
  ];

  dependencies = [
    meshcat
  ];

  optional-dependencies = {
    doc = [
      sphinx
    ];
  };

  pythonImportsCheck = [
    "meshcat_shapes"
  ];

  meta = {
    description = "Additional shapes to decorate MeshCat scenes (frames, text";
    homepage = "https://github.com/stephane-caron/meshcat-shapes";
    changelog = "https://github.com/stephane-caron/meshcat-shapes/blob/${src.rev}/CHANGELOG.md";
    license = lib.licenses.asl20;
    maintainers = with lib.maintainers; [ nim65s ];
  };
}
