"use client";

import { useMemo, useState } from "react";
import { Activity, Goal, Shield, TrendingUp, Zap } from "lucide-react";

type RushFeatures = {
  nearest_support_dist: number;
  second_support_dist: number;
  teammates_in_radius: number;
  lane_spread: number;
  max_lane_width: number;
  lane_balance: number;
  depth_range: number;
  depth_variance: number;
  is_flat_line: number;
  mean_team_speed: number;
  speed_variance: number;
  carrier_speed: number;
  carrier_acceleration: number;
  nearest_defender_dist: number;
  defender_closing_speed: number;
  defenders_between: number;
};

type PredictionResponse = {
  shot_probability: number;
  controlled_entry_probability: number;
  xg_15s: number;
};

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

const defaultFeatures: RushFeatures = {
  nearest_support_dist: 19.9,
  second_support_dist: 28,
  teammates_in_radius: 1,
  lane_spread: 20,
  max_lane_width: 40,
  lane_balance: 5,
  depth_range: 10,
  depth_variance: 15,
  is_flat_line: 0,
  mean_team_speed: 25,
  speed_variance: 5,
  carrier_speed: 25,
  carrier_acceleration: 0,
  nearest_defender_dist: 10,
  defender_closing_speed: 5,
  defenders_between: 2,
};

const featureGroups: Array<{
  title: string;
  accent: string;
  fields: Array<{ key: keyof RushFeatures; label: string; step?: number }>;
}> = [
  {
    title: "Support Structure",
    accent: "from-cyan-400/30 to-blue-500/10",
    fields: [
      { key: "nearest_support_dist", label: "Nearest Support Distance", step: 0.1 },
      { key: "second_support_dist", label: "Second Support Distance", step: 0.1 },
      { key: "teammates_in_radius", label: "Teammates In Radius", step: 1 },
    ],
  },
  {
    title: "Lane Geometry",
    accent: "from-sky-400/30 to-indigo-500/10",
    fields: [
      { key: "lane_spread", label: "Lane Spread", step: 0.1 },
      { key: "max_lane_width", label: "Max Lane Width", step: 0.1 },
      { key: "lane_balance", label: "Lane Balance", step: 0.1 },
    ],
  },
  {
    title: "Depth & Alignment",
    accent: "from-blue-400/30 to-cyan-500/10",
    fields: [
      { key: "depth_range", label: "Depth Range", step: 0.1 },
      { key: "depth_variance", label: "Depth Variance", step: 0.1 },
      { key: "is_flat_line", label: "Flat Line (0 or 1)", step: 1 },
    ],
  },
  {
    title: "Speed Profile",
    accent: "from-emerald-400/30 to-teal-500/10",
    fields: [
      { key: "mean_team_speed", label: "Mean Team Speed", step: 0.1 },
      { key: "speed_variance", label: "Speed Variance", step: 0.1 },
      { key: "carrier_speed", label: "Carrier Speed", step: 0.1 },
      { key: "carrier_acceleration", label: "Carrier Acceleration", step: 0.1 },
    ],
  },
  {
    title: "Defensive Pressure",
    accent: "from-rose-400/25 to-orange-500/10",
    fields: [
      { key: "nearest_defender_dist", label: "Nearest Defender Distance", step: 0.1 },
      { key: "defender_closing_speed", label: "Defender Closing Speed", step: 0.1 },
      { key: "defenders_between", label: "Defenders Between", step: 1 },
    ],
  },
];

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function dangerLabel(xg: number) {
  if (xg >= 0.2) return "High danger";
  if (xg >= 0.08) return "Moderate danger";
  return "Low danger";
}

function StatCard({
  title,
  value,
  subtitle,
  icon,
}: {
  title: string;
  value: string;
  subtitle: string;
  icon: React.ReactNode;
}) {
  return (
    <div className="relative overflow-hidden rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl backdrop-blur-sm">
      <div className="absolute inset-0 bg-gradient-to-br from-white/10 via-transparent to-transparent" />
      <div className="relative flex items-start justify-between gap-4">
        <div>
          <div className="text-xs uppercase tracking-[0.25em] text-slate-400">{title}</div>
          <div className="mt-3 text-3xl font-bold text-white md:text-4xl">{value}</div>
          <div className="mt-2 text-sm text-slate-300">{subtitle}</div>
        </div>
        <div className="rounded-2xl border border-cyan-300/20 bg-cyan-300/10 p-3 text-cyan-200">
          {icon}
        </div>
      </div>
    </div>
  );
}

function FeatureInput({
  label,
  value,
  onChange,
  step = 0.1,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  step?: number;
}) {
  return (
    <label className="space-y-2">
      <div className="flex items-center justify-between text-sm text-slate-200">
        <span>{label}</span>
        <span className="rounded-full bg-slate-900/70 px-2 py-0.5 text-xs text-cyan-200">{value}</span>
      </div>
      <input
        type="range"
        min={0}
        max={step === 1 ? 10 : 60}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-2 w-full cursor-pointer appearance-none rounded-full bg-slate-700 accent-cyan-400"
      />
      <input
        type="number"
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full rounded-2xl border border-white/10 bg-slate-950/70 px-3 py-2 text-sm text-white outline-none ring-0 placeholder:text-slate-500"
      />
    </label>
  );
}

export default function HockeyRushDashboard() {
  const [features, setFeatures] = useState<RushFeatures>(defaultFeatures);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const summary = useMemo(() => {
    if (!result) return null;
    if (result.shot_probability > 0.65) return "Rush profile suggests an immediate shot threat.";
    if (result.controlled_entry_probability > 0.6) return "Structure supports a clean offensive-zone entry.";
    return "Current rush shape looks more transitional than dangerous.";
  }, [result]);

  async function runPrediction() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(features),
      });

      if (!res.ok) {
        throw new Error("Prediction request failed.");
      }

      const data: PredictionResponse = await res.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  function updateField<K extends keyof RushFeatures>(key: K, value: RushFeatures[K]) {
    setFeatures((prev) => ({ ...prev, [key]: value }));
  }

  function loadPreset(type: "speed" | "support" | "spread") {
    if (type === "speed") {
      setFeatures((prev) => ({
        ...prev,
        mean_team_speed: 31,
        carrier_speed: 33,
        carrier_acceleration: 6,
        nearest_support_dist: 15,
      }));
    }
    if (type === "support") {
      setFeatures((prev) => ({
        ...prev,
        nearest_support_dist: 12,
        second_support_dist: 20,
        teammates_in_radius: 2,
        defenders_between: 1,
      }));
    }
    if (type === "spread") {
      setFeatures((prev) => ({
        ...prev,
        lane_spread: 28,
        max_lane_width: 48,
        lane_balance: 3,
        depth_range: 16,
      }));
    }
  }

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top,_rgba(34,211,238,0.18),_transparent_32%),linear-gradient(180deg,_#020617_0%,_#061327_40%,_#020617_100%)] text-white">
      <div className="mx-auto max-w-7xl px-4 py-8 md:px-8 md:py-10">
        <section className="relative overflow-hidden rounded-[32px] border border-cyan-300/15 bg-white/5 p-6 shadow-2xl backdrop-blur md:p-8">
          <div className="absolute inset-0 bg-[linear-gradient(120deg,rgba(125,211,252,0.08),transparent_35%,rgba(255,255,255,0.03)_60%,transparent_100%)]" />
          <div className="relative grid gap-8 lg:grid-cols-[1.2fr_0.8fr]">
            <div>
              <div className="inline-flex items-center gap-2 rounded-full border border-cyan-300/20 bg-cyan-300/10 px-3 py-1 text-xs uppercase tracking-[0.3em] text-cyan-200">
                <Shield className="h-3.5 w-3.5" />
                Hockey Transition Intelligence
              </div>
              <h1 className="mt-4 text-4xl font-black tracking-tight text-white md:text-6xl">
                Rush Structure
                <span className="block bg-gradient-to-r from-cyan-300 via-sky-200 to-white bg-clip-text text-transparent">
                  Command Center
                </span>
              </h1>
              <p className="mt-4 max-w-2xl text-base leading-7 text-slate-300 md:text-lg">
                Evaluate how support shape, lane geometry, speed, and pressure influence shot threat,
                controlled entry probability, and 15-second offensive danger.
              </p>

              <div className="mt-6 flex flex-wrap gap-3">
                <button
                  onClick={() => loadPreset("speed")}
                  className="rounded-2xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-100 transition hover:bg-cyan-300/10"
                >
                  High-Speed Rush
                </button>
                <button
                  onClick={() => loadPreset("support")}
                  className="rounded-2xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-100 transition hover:bg-cyan-300/10"
                >
                  Tight Support
                </button>
                <button
                  onClick={() => loadPreset("spread")}
                  className="rounded-2xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-slate-100 transition hover:bg-cyan-300/10"
                >
                  Wide-Lane Entry
                </button>
              </div>
            </div>

            <div className="relative rounded-[28px] border border-white/10 bg-slate-950/60 p-5">
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <div className="text-xs uppercase tracking-[0.25em] text-slate-400">Live Scouting Board</div>
                  <div className="mt-1 text-lg font-semibold text-white">Rush Snapshot</div>
                </div>
                <div className="rounded-2xl border border-cyan-300/20 bg-cyan-300/10 px-3 py-1 text-xs text-cyan-200">
                  API Ready
                </div>
              </div>
              <div className="space-y-3 text-sm text-slate-300">
                <div className="flex items-center justify-between rounded-2xl bg-white/5 px-4 py-3">
                  <span>Support Spacing</span>
                  <span className="font-semibold text-white">{features.nearest_support_dist.toFixed(1)} ft</span>
                </div>
                <div className="flex items-center justify-between rounded-2xl bg-white/5 px-4 py-3">
                  <span>Lane Spread</span>
                  <span className="font-semibold text-white">{features.lane_spread.toFixed(1)}</span>
                </div>
                <div className="flex items-center justify-between rounded-2xl bg-white/5 px-4 py-3">
                  <span>Carrier Speed</span>
                  <span className="font-semibold text-white">{features.carrier_speed.toFixed(1)}</span>
                </div>
                <div className="flex items-center justify-between rounded-2xl bg-white/5 px-4 py-3">
                  <span>Defenders Between</span>
                  <span className="font-semibold text-white">{features.defenders_between.toFixed(0)}</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="mt-8 grid gap-4 md:grid-cols-3">
          <StatCard
            title="Shot Threat"
            value={result ? formatPercent(result.shot_probability) : "--"}
            subtitle="Probability of a shot within 10 seconds"
            icon={<Goal className="h-6 w-6" />}
          />
          <StatCard
            title="Entry Success"
            value={result ? formatPercent(result.controlled_entry_probability) : "--"}
            subtitle="Probability of controlled offensive-zone entry"
            icon={<TrendingUp className="h-6 w-6" />}
          />
          <StatCard
            title="Offensive Danger"
            value={result ? result.xg_15s.toFixed(3) : "--"}
            subtitle={result ? dangerLabel(result.xg_15s) : "Expected goals over 15 seconds"}
            icon={<Zap className="h-6 w-6" />}
          />
        </section>

        <section className="mt-8 grid gap-8 lg:grid-cols-[1.2fr_0.8fr]">
          <div className="rounded-[28px] border border-white/10 bg-white/5 p-5 shadow-2xl backdrop-blur md:p-6">
            <div className="mb-6 flex items-center justify-between gap-4">
              <div>
                <h2 className="text-2xl font-bold text-white">Rush Configuration</h2>
                <p className="mt-1 text-sm text-slate-400">Tune your structure like a live bench-side analytics tool.</p>
              </div>
              <button
                onClick={runPrediction}
                disabled={loading}
                className="inline-flex items-center gap-2 rounded-2xl bg-cyan-400 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <Activity className="h-4 w-4" />
                {loading ? "Running Model..." : "Run Rush Analysis"}
              </button>
            </div>

            <div className="grid gap-5">
              {featureGroups.map((group) => (
                <div
                  key={group.title}
                  className={`rounded-[24px] border border-white/10 bg-gradient-to-br ${group.accent} p-4 md:p-5`}
                >
                  <div className="mb-4 text-lg font-semibold text-white">{group.title}</div>
                  <div className="grid gap-4 md:grid-cols-2">
                    {group.fields.map((field) => (
                      <FeatureInput
                        key={field.key}
                        label={field.label}
                        value={features[field.key]}
                        step={field.step}
                        onChange={(value) => updateField(field.key, value)}
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="space-y-6">
            <div className="rounded-[28px] border border-white/10 bg-slate-950/60 p-6 shadow-2xl">
              <div className="text-xs uppercase tracking-[0.25em] text-slate-400">Scouting Readout</div>
              <div className="mt-3 text-2xl font-bold text-white">Analyst Summary</div>
              <p className="mt-4 text-sm leading-7 text-slate-300">
                {summary ?? "Run a prediction to generate a live summary of this transition rush profile."}
              </p>
              {error && (
                <div className="mt-4 rounded-2xl border border-rose-400/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
                  {error}
                </div>
              )}
            </div>

            <div className="rounded-[28px] border border-white/10 bg-white/5 p-6 shadow-2xl backdrop-blur">
              <div className="text-xs uppercase tracking-[0.25em] text-slate-400">Quick Interpretation</div>
              <div className="mt-4 space-y-3 text-sm text-slate-300">
                <div className="rounded-2xl bg-white/5 px-4 py-3">
                  <span className="font-semibold text-white">Tighter support</span> generally helps create short passing options.
                </div>
                <div className="rounded-2xl bg-white/5 px-4 py-3">
                  <span className="font-semibold text-white">Wider lanes</span> can stretch defenders and improve entry geometry.
                </div>
                <div className="rounded-2xl bg-white/5 px-4 py-3">
                  <span className="font-semibold text-white">Higher speed</span> often increases transition pressure before the defense resets.
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
