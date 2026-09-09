// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file dgTwoTrackSelection.cxx
/// \brief Compact two-track double-gap candidate table. Requires o2-analysis-event-selection-service
/// \author Nazar Burmasov

#include "Common/CCDB/EventSelectionParams.h"
#include "Common/CCDB/RCTSelectionFlags.h"
#include "Common/Core/trackUtilities.h"
#include "Common/DataModel/EventSelection.h"

#include <CCDB/BasicCCDBManager.h>
#include <CommonConstants/LHCConstants.h>
#include <CommonConstants/PhysicsConstants.h>
#include <DataFormatsFIT/Triggers.h>
#include <DataFormatsParameters/GRPLHCIFData.h>
#include <DataFormatsParameters/GRPMagField.h>
#include <DetectorsBase/MatLayerCylSet.h>
#include <DetectorsBase/Propagator.h>
#include <Framework/AnalysisTask.h>
#include <Framework/Configurable.h>
#include <Framework/HistogramRegistry.h>
#include <Framework/InitContext.h>
#include <Framework/Logger.h>
#include <Framework/runDataProcessing.h>

#include <TH1.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

using namespace o2;
using namespace o2::framework;

namespace
{
constexpr int NBCsPerOrbit = o2::constants::lhc::LHCMaxBunches;
constexpr int MaxStoredDistance = 15;
constexpr int TVXTriggerBit = o2::fit::Triggers::bitVertex;
constexpr float InvLightSpeedCm = 1.f / 29.9792458f;
constexpr float InvTrackTimeRes = 1.f / 0.199997f;
constexpr float MaxFITTime = 30.f;
constexpr std::array<float, 9> TrackingMassOverZ{o2::constants::physics::MassElectron,
                                                 o2::constants::physics::MassMuon,
                                                 o2::constants::physics::MassPionCharged,
                                                 o2::constants::physics::MassKaonCharged,
                                                 o2::constants::physics::MassProton,
                                                 o2::constants::physics::MassDeuteron,
                                                 o2::constants::physics::MassTriton,
                                                 o2::constants::physics::MassHelium3 / 2.f,
                                                 o2::constants::physics::MassAlpha / 2.f};

constexpr std::array<const char*, 23> RCTLabels{"CPV bad", "EMC bad", "EMC limited acceptance", "FDD bad", "FT0 bad", "FV0 bad",
                                                "HMP bad", "ITS bad", "ITS limited acceptance", "MCH bad", "MCH limited acceptance",
                                                "MFT bad", "MFT limited acceptance", "MID bad", "MID limited acceptance", "PHS bad",
                                                "TOF bad", "TOF limited acceptance", "TPC bad tracking", "TPC bad PID",
                                                "TPC limited acceptance", "TRD bad", "ZDC bad"};

constexpr float MassPi2 = o2::constants::physics::MassPionCharged * o2::constants::physics::MassPionCharged;
constexpr float MassKa2 = o2::constants::physics::MassKaonCharged * o2::constants::physics::MassKaonCharged;
constexpr float MassPr2 = o2::constants::physics::MassProton * o2::constants::physics::MassProton;

template <typename TTrack>
float getTOFNSigma(TTrack const& track, float mass2)
{
  if (!track.hasTOF()) {
    return -999.f;
  }
  const float p2 = track.pt() * track.pt() + track.pz() * track.pz();
  if (p2 <= 0.f) {
    return -999.f;
  }
  const auto pid = track.pidForTracking();
  const float massOverZ = pid < TrackingMassOverZ.size() ? TrackingMassOverZ[pid] : 0.f;
  const float factor = track.length() * InvLightSpeedCm;
  const float expectedTracking = factor * std::sqrt(1.f + massOverZ * massOverZ / p2);
  const float expectedHypothesis = factor * std::sqrt(1.f + mass2 / p2);
  return (track.trackTime() + expectedTracking - expectedHypothesis) * InvTrackTimeRes;
}

struct ForwardDistances {
  int ft0;
  int fv0;
  int fdd;
};
} // namespace

namespace o2::aod::dg_two_track
{
DECLARE_SOA_COLUMN(SourceCollisionId, sourceCollisionId, int64_t);
DECLARE_SOA_COLUMN(RunNumber, runNumber, int32_t);
DECLARE_SOA_COLUMN(GlobalBC, globalBC, uint64_t);
DECLARE_SOA_COLUMN(Timestamp, timestamp, uint64_t);
DECLARE_SOA_COLUMN(PosX, posX, float);
DECLARE_SOA_COLUMN(PosY, posY, float);
DECLARE_SOA_COLUMN(PosZ, posZ, float);
DECLARE_SOA_COLUMN(Px, px, std::vector<float>);
DECLARE_SOA_COLUMN(Py, py, std::vector<float>);
DECLARE_SOA_COLUMN(Pz, pz, std::vector<float>);
DECLARE_SOA_COLUMN(TPCSignal, tpcSignal, std::vector<float>);
DECLARE_SOA_COLUMN(TOFNSigmaPi, tofNSigmaPi, std::vector<float>);
DECLARE_SOA_COLUMN(TOFNSigmaKa, tofNSigmaKa, std::vector<float>);
DECLARE_SOA_COLUMN(TOFNSigmaPr, tofNSigmaPr, std::vector<float>);
DECLARE_SOA_COLUMN(ITSClusterSizes, itsClusterSizes, std::vector<uint32_t>);
DECLARE_SOA_COLUMN(NClusters, nClusters, std::vector<uint8_t>);
DECLARE_SOA_COLUMN(Sign, sign, std::vector<int8_t>);
DECLARE_SOA_COLUMN(MinimumDistanceB, minimumDistanceB, int8_t);
DECLARE_SOA_COLUMN(MinimumDistanceFT0, minimumDistanceFT0, int8_t);
DECLARE_SOA_COLUMN(MinimumDistanceFV0, minimumDistanceFV0, int8_t);
DECLARE_SOA_COLUMN(MinimumDistanceFDD, minimumDistanceFDD, int8_t);
DECLARE_SOA_COLUMN(RctFlags, rctFlags, uint32_t);
DECLARE_SOA_COLUMN(RctSelection, rctSelection, uint8_t);
} // namespace o2::aod::dg_two_track

namespace o2::aod
{
DECLARE_SOA_TABLE(DGTwoTrackSels, "AOD", "DGTWOTRACKSEL",
                  dg_two_track::SourceCollisionId,
                  dg_two_track::RunNumber,
                  dg_two_track::GlobalBC,
                  dg_two_track::Timestamp,
                  dg_two_track::PosX,
                  dg_two_track::PosY,
                  dg_two_track::PosZ,
                  dg_two_track::Px,
                  dg_two_track::Py,
                  dg_two_track::Pz,
                  dg_two_track::TPCSignal,
                  dg_two_track::TOFNSigmaPi,
                  dg_two_track::TOFNSigmaKa,
                  dg_two_track::TOFNSigmaPr,
                  dg_two_track::ITSClusterSizes,
                  dg_two_track::NClusters,
                  dg_two_track::Sign,
                  dg_two_track::MinimumDistanceB,
                  dg_two_track::MinimumDistanceFT0,
                  dg_two_track::MinimumDistanceFV0,
                  dg_two_track::MinimumDistanceFDD,
                  dg_two_track::RctFlags,
                  dg_two_track::RctSelection);
} // namespace o2::aod

struct DGTwoTrackSelection {
  Produces<aod::DGTwoTrackSels> selectedEvents;
  HistogramRegistry registry{"registry", {}};

  Configurable<float> maxAbsEta{"maxAbsEta", 0.8f, "Maximum |eta| of selected tracks"};
  Configurable<float> minPt{"minPt", 0.2f, "Minimum pT of selected tracks (GeV/c)"};
  Configurable<int> rctMode{"rctMode", 1, "RCT: 0=off, 1=UD, 2=UD+TOF"};
  Configurable<int> vetoBCWindow{"vetoBCWindow", 0, "FIT veto half-window in BC; <0 disables"};
  Configurable<int> vetoFT0{"vetoFT0", 1, "FT0 veto: 0=off, 1=on"};
  Configurable<int> vetoFV0{"vetoFV0", 0, "FV0 veto: 0=off, 1=on"};
  Configurable<int> vetoFDD{"vetoFDD", 0, "FDD veto: 0=off, 1=on"};
  Configurable<std::string> ccdbURL{"ccdbURL", "http://alice-ccdb.cern.ch", "CCDB URL"};

  using Collisions = soa::Join<aod::Collisions, aod::EvSels>;
  using BCs = soa::Join<aod::BCsWithTimestamps, aod::BcSels>;
  using Tracks = soa::Join<aod::TracksIU, aod::TracksExtra>;

  Service<o2::ccdb::BasicCCDBManager> ccdb;
  o2::base::MatLayerCylSet* lut = nullptr;
  int currentRun = -1;
  std::array<int8_t, NBCsPerOrbit> minimumDistanceB{};

  o2::aod::rctsel::RCTFlagsChecker rctChecker{o2::aod::rctsel::kFDDBad, o2::aod::rctsel::kFT0Bad, o2::aod::rctsel::kFV0Bad,
                                              o2::aod::rctsel::kITSBad, o2::aod::rctsel::kTPCBadTracking, o2::aod::rctsel::kTPCBadPID,
                                              o2::aod::rctsel::kCcdbObjectLoaded};
  o2::aod::rctsel::RCTFlagsChecker rctCheckerWithTOF{o2::aod::rctsel::kFDDBad, o2::aod::rctsel::kFT0Bad, o2::aod::rctsel::kFV0Bad,
                                                     o2::aod::rctsel::kITSBad, o2::aod::rctsel::kTPCBadTracking, o2::aod::rctsel::kTPCBadPID,
                                                     o2::aod::rctsel::kTOFBad, o2::aod::rctsel::kCcdbObjectLoaded};

  void init(InitContext const&)
  {
    if (rctMode.value < 0 || rctMode.value > 2) {
      LOGF(fatal, "rctMode must be 0, 1 or 2");
    }
    ccdb->setURL(ccdbURL.value);
    ccdb->setCaching(true);
    ccdb->setLocalObjectValidityChecking();
    registry.add("hTVX", "TVX counts per run;run;TVX BCs", HistType::kTH1D, {{1, 0.5, 1.5}});
    registry.get<TH1>(HIST("hTVX"))->SetCanExtend(TH1::kXaxis);
    registry.add("hRCTFlags", "RCT flags for TVX;RCT flag;TVX counts", HistType::kTH1D, {{32, -0.5, 31.5}});
    auto hRCT = registry.get<TH1>(HIST("hRCTFlags"));
    for (size_t i = 0; i < RCTLabels.size(); ++i) {
      hRCT->GetXaxis()->SetBinLabel(i + 1, RCTLabels[i]);
    }
    hRCT->GetXaxis()->SetBinLabel(o2::aod::rctsel::kCcdbObjectLoaded + 1, "RCT object unavailable");
    LOGF(info, "Veto config: window=%d FT0=%d FV0=%d FDD=%d", vetoBCWindow.value, vetoFT0.value, vetoFV0.value, vetoFDD.value);
  }

  template <typename TBC>
  void updateRun(TBC const& bc)
  {
    if (bc.runNumber() == currentRun) {
      return;
    }
    auto* field = ccdb->getForTimeStamp<o2::parameters::GRPMagField>("GLO/Config/GRPMagField", bc.timestamp());
    auto* lhcif = ccdb->getForTimeStamp<o2::parameters::GRPLHCIFData>("GLO/Config/GRPLHCIF", bc.timestamp());
    lut = o2::base::MatLayerCylSet::rectifyPtrFromFile(
      ccdb->getForTimeStamp<o2::base::MatLayerCylSet>("GLO/Param/MatLUT", bc.timestamp()));
    if (!field || !lhcif || !lut) {
      LOGF(fatal, "Could not load run-dependent CCDB objects for run %d", bc.runNumber());
    }
    o2::base::Propagator::initFieldFromGRP(field);
    o2::base::Propagator::Instance()->setMatLUT(lut);
    const auto bunchPattern = lhcif->getBunchFilling().getBCPattern();
    minimumDistanceB.fill(MaxStoredDistance + 1);
    for (int bcInOrbit = 0; bcInOrbit < NBCsPerOrbit; ++bcInOrbit) {
      if (bunchPattern[bcInOrbit]) {
        minimumDistanceB[bcInOrbit] = 0;
        continue;
      }
      for (int distance = 1; distance <= MaxStoredDistance; ++distance) {
        const int previous = (bcInOrbit - distance + NBCsPerOrbit) % NBCsPerOrbit;
        const int next = (bcInOrbit + distance) % NBCsPerOrbit;
        if (bunchPattern[previous] || bunchPattern[next]) {
          minimumDistanceB[bcInOrbit] = distance;
          break;
        }
      }
    }
    currentRun = bc.runNumber();
  }

  int getMinimumDistance(uint64_t globalBC, std::vector<int64_t> const& activeBCs, int maxDistance) const
  {
    auto it = std::lower_bound(activeBCs.begin(), activeBCs.end(), static_cast<int64_t>(globalBC));
    int64_t distance = maxDistance + 1;
    if (it != activeBCs.end()) {
      distance = std::min(distance, std::abs(*it - static_cast<int64_t>(globalBC)));
    }
    if (it != activeBCs.begin()) {
      distance = std::min(distance, std::abs(*(it - 1) - static_cast<int64_t>(globalBC)));
    }
    return std::min<int64_t>(distance, maxDistance + 1);
  }

  template <typename T>
  bool isGoodRCT(T const& row)
  {
    return rctMode.value == 0 || (rctMode.value == 1 ? rctChecker(row) : rctCheckerWithTOF(row));
  }

  template <typename T>
  uint8_t getRCTSelection(T const& row)
  {
    return (rctChecker(row) ? 1u : 0u) | (rctCheckerWithTOF(row) ? 2u : 0u);
  }

  bool hasForwardVeto(ForwardDistances const& distance)
  {
    const int window = vetoBCWindow.value;
    return window >= 0 && (((vetoFT0.value != 0) && distance.ft0 <= window) || ((vetoFV0.value != 0) && distance.fv0 <= window) || ((vetoFDD.value != 0) && distance.fdd <= window));
  }

  int8_t getStoredDistance(int distance) const { return std::min(distance, MaxStoredDistance + 1); }

  void process(Collisions const& collisions, Tracks const& tracks, BCs const& bcs, aod::FT0s const& ft0s, aod::FV0As const& fv0s, aod::FDDs const& fdds)
  {
    auto hTVX = registry.get<TH1>(HIST("hTVX"));

    updateRun(bcs.begin());

    std::vector<int64_t> bcsWithFT0;
    std::vector<int64_t> bcsWithFV0;
    std::vector<int64_t> bcsWithFDD;
    bcsWithFT0.reserve(ft0s.size());
    bcsWithFV0.reserve(fv0s.size());
    bcsWithFDD.reserve(fdds.size());

    for (const auto& ft0 : ft0s) {
      const auto bc = ft0.bc_as<BCs>();
      const auto gbc = bc.globalBC();
      if (ft0.timeA() < MaxFITTime || ft0.timeC() < MaxFITTime) {
        bcsWithFT0.push_back(gbc);
      }
      if (!bc.selection_bit(aod::evsel::kNoTimeFrameBorder) || minimumDistanceB[gbc % NBCsPerOrbit] != 0 || ((ft0.triggerMask() & BIT(TVXTriggerBit)) == 0u)) {
        continue;
      }
      const uint32_t flags = bc.rct_raw();
      for (int bit = 0; bit < 32; ++bit) {
        if ((flags & BIT(bit)) != 0u) {
          registry.fill(HIST("hRCTFlags"), bit);
        }
      }
      if (isGoodRCT(bc)) {
        hTVX->Fill(std::to_string(bc.runNumber()).c_str(), 1.);
      }
    }
    for (const auto& fv0 : fv0s) {
      if (fv0.time() < MaxFITTime) {
        bcsWithFV0.push_back(fv0.bc_as<BCs>().globalBC());
      }
    }
    for (const auto& fdd : fdds) {
      if (fdd.timeA() < MaxFITTime || fdd.timeC() < MaxFITTime) {
        bcsWithFDD.push_back(fdd.bc_as<BCs>().globalBC());
      }
    }

    const int64_t collisionOffset = (collisions.size() != 0) ? collisions.begin().globalIndex() : 0;
    std::vector<std::vector<int64_t>> tracksByCollision(collisions.size());
    for (const auto& track : tracks) {
      const int64_t index = track.collisionId() - collisionOffset;
      if (index >= 0 && index < static_cast<int64_t>(tracksByCollision.size())) {
        tracksByCollision[index].push_back(track.globalIndex());
      }
    }

    for (const auto& collision : collisions) {
      if (collision.numContrib() != 2) {
        continue;
      }
      auto bc = collision.bc_as<BCs>();
      if (!bc.selection_bit(aod::evsel::kNoTimeFrameBorder) || !isGoodRCT(bc)) {
        continue;
      }
      const int scanDistance = std::max(MaxStoredDistance, std::max(vetoBCWindow.value, 0));
      const ForwardDistances forwardDistance{.ft0 = getMinimumDistance(bc.globalBC(), bcsWithFT0, scanDistance),
                                             .fv0 = getMinimumDistance(bc.globalBC(), bcsWithFV0, scanDistance),
                                             .fdd = getMinimumDistance(bc.globalBC(), bcsWithFDD, scanDistance)};
      if (hasForwardVeto(forwardDistance)) {
        continue;
      }
      std::vector<float> px, py, pz, tpcSignal, tofNSigmaPi, tofNSigmaKa, tofNSigmaPr;
      std::vector<uint32_t> itsClusterSizes;
      std::vector<uint8_t> nClusters;
      std::vector<int8_t> sign;
      for (const auto trackIndex : tracksByCollision[collision.globalIndex() - collisionOffset]) {
        const auto track = tracks.iteratorAt(trackIndex);
        if (!track.isPVContributor() || !track.hasITS() || !track.hasTPC()) {
          continue;
        }
        auto par = getTrackPar(track);
        std::array<float, 2> dca{999.f, 999.f};
        o2::base::Propagator::Instance()->propagateToDCABxByBz(
          {collision.posX(), collision.posY(), collision.posZ()},
          par, 2.f, o2::base::Propagator::MatCorrType::USEMatCorrLUT, &dca);
        if (std::abs(par.getEta()) > maxAbsEta.value || par.getPt() < minPt.value) {
          continue;
        }
        if (px.size() >= 2) {
          break;
        }
        const float pt = par.getPt();
        px.push_back(pt * std::cos(par.getPhi()));
        py.push_back(pt * std::sin(par.getPhi()));
        pz.push_back(pt * par.getTgl());
        tpcSignal.push_back(track.tpcSignal());
        tofNSigmaPi.push_back(getTOFNSigma(track, MassPi2));
        tofNSigmaKa.push_back(getTOFNSigma(track, MassKa2));
        tofNSigmaPr.push_back(getTOFNSigma(track, MassPr2));
        itsClusterSizes.push_back(track.itsClusterSizes());
        nClusters.push_back(track.tpcNClsFound());
        sign.push_back(track.sign());
      }
      if (px.size() != 2) {
        continue;
      }
      const int bcInOrbit = bc.globalBC() % NBCsPerOrbit;
      selectedEvents(collision.globalIndex(), bc.runNumber(), bc.globalBC(), bc.timestamp(),
                     collision.posX(), collision.posY(), collision.posZ(),
                     px, py, pz, tpcSignal, tofNSigmaPi, tofNSigmaKa, tofNSigmaPr, itsClusterSizes, nClusters, sign,
                     minimumDistanceB[bcInOrbit], getStoredDistance(forwardDistance.ft0), getStoredDistance(forwardDistance.fv0), getStoredDistance(forwardDistance.fdd),
                     static_cast<uint32_t>(bc.rct_raw()), getRCTSelection(bc));
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc) { return WorkflowSpec{adaptAnalysisTask<DGTwoTrackSelection>(cfgc)}; }
