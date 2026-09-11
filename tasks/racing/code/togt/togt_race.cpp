// Copied from the crazy_track research repo (scripts/togt/togt_race.cpp @ 58eed32) for
// Lesson 6 of the racing task; built by build.sh in this folder.
//
// Minimal driver for TOGT-Planner (github.com/FSC-Lab/TOGT-Planner): plan a race
// trajectory from a parameter set + track YAML and save the sampled trajectory as
// CSV (t, p, q, v, w, a_lin, ..., u_1..4). crazy_track loads that CSV through
// trajectories/sampled.py and races it with a learned tracker.
//
//   togt_race <params_dir> <setups.yaml> <track.yaml> <out.csv> <togt|aos> [out_wpt.yaml]
//
//   togt : RacePlanner::planTOGT  (TOGT only; uses the *init* params)
//   aos  : RacePlanner::plan      (AOS-TOGT, "true time optimality": init then refine)
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include "drolib/race/race_params.hpp"
#include "drolib/race/race_planner.hpp"
#include "drolib/race/race_track.hpp"

using namespace drolib;

int main(int argc, char** argv) {
  if (argc < 6) {
    std::cerr << "usage: togt_race <params_dir> <setups.yaml> <track.yaml> <out.csv> "
                 "<togt|aos> [out_wpt.yaml]\n";
    return 2;
  }
  // TOGT's loader joins <params_dir>/<file> and then checkFile() prefixes
  // <params_dir> AGAIN, so a relative params_dir is looked up twice and fails
  // ("Main configuration file not found"), after which planning segfaults on
  // empty parameters. Absolutize everything up front.
  const std::filesystem::path params_dir = std::filesystem::absolute(argv[1]);
  const std::string setups(argv[2]);
  const std::filesystem::path track_path = std::filesystem::absolute(argv[3]);
  const std::filesystem::path out_csv = std::filesystem::absolute(argv[4]);
  const std::string mode(argv[5]);

  auto params = std::make_shared<RaceParams>(params_dir.string(), setups);
  auto planner = std::make_shared<RacePlanner>(*params);
  auto track = std::make_shared<RaceTrack>(track_path);

  const bool ok = (mode == "aos") ? planner->plan(track) : planner->planTOGT(track);
  if (!ok) {
    std::cerr << "planning FAILED (mode=" << mode << ")\n";
    return 1;
  }
  std::cout << planner->getExtremum() << std::endl;
  MincoSnapTrajectory traj = planner->getTrajectory();
  if (!traj.save(out_csv.string())) {
    std::cerr << "save FAILED: " << out_csv << "\n";
    return 1;
  }
  if (argc > 6) traj.saveSegments(argv[6], params->tprefine.piecesPerSegment);
  std::cout << "saved " << out_csv << std::endl;
  return 0;
}
