import React, { useEffect, useMemo, useState } from "react";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { ApiError, evaluateFischerPgn, getHeldoutExamples, getOpeningStats } from "../services/api.js";
import GlassPanel from "./GlassPanel.jsx";

const CHART_COLORS = ["#e4c257", "#b3543f", "#7a9d6e", "#6e7681", "#a57c4b", "#8670aa"];

/**
 * Display evidence of Fischer-style policy learning through openings and PGN agreement.
 *
 * @returns {React.JSX.Element} Style verification dashboard.
 */
export default function Dashboard({ showHeatmap, onHeatmapChange, fischerColor }) {
  const [openingStats, setOpeningStats] = useState(null);
  const [heldoutExamples, setHeldoutExamples] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    setIsLoading(true);
    setError("");
    Promise.all([
      getOpeningStats(fischerColor === "b" ? "black" : "white"),
      getHeldoutExamples(),
    ])
      .then(([stats, examples]) => {
        if (!active) return;
        setOpeningStats(stats);
        setHeldoutExamples(examples);
      })
      .catch((err) => active && setError(err instanceof ApiError ? err.message : "Analytics are unavailable."))
      .finally(() => active && setIsLoading(false));
    return () => {
      active = false;
    };
  }, [fischerColor]);

  const charts = useMemo(
    () => {
      const isFischerBlack = fischerColor === "b";
      return [
        {
          title: isFischerBlack
            ? "Fischer defense"
            : "Fischer opening",
          data: openingStats?.actual ?? [],
        },
        {
          title: isFischerBlack
            ? "AI defensive policy"
            : "AI opening policy",
          data: openingStats?.ai ?? [],
        },
      ];
    },
    [fischerColor, openingStats]
  );

  const handlePgnUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setIsEvaluating(true);
    setError("");
    try {
      setMetrics(await evaluateFischerPgn(file));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "PGN evaluation failed.");
    } finally {
      setIsEvaluating(false);
      event.target.value = "";
    }
  };

  return (
    <GlassPanel style={{ width: "min(88vw, 480px)", padding: "1.7rem", display: "flex", flexDirection: "column", gap: "1.4rem" }}>
      <div>
        <p style={eyebrowStyle}>Style verification</p>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.65rem" }}>
        <StatCard label="Top-1 accuracy" value={metrics ? `${metrics.top1_match_rate.toFixed(2)}%` : "—"} />
        <StatCard label="Top-3 accuracy" value={metrics ? `${metrics.top3_match_rate.toFixed(2)}%` : "—"} />
      </div>
      <label style={uploadLabelStyle}>
        <span>{isEvaluating ? "Evaluating PGN…" : "Evaluate external held-out PGN"}</span>
        <input type="file" accept=".pgn,application/x-chess-pgn" onChange={handlePgnUpload} disabled={isEvaluating} hidden />
      </label>
      {metrics && <span style={captionStyle}>{metrics.positions_evaluated} Fischer moves across {metrics.games_evaluated} games</span>}
      <label style={heatmapLabelStyle}>
        <input
          type="checkbox"
          checked={showHeatmap}
          onChange={(event) => onHeatmapChange(event.target.checked)}
        />
        <span>Show AI activity heatmap</span>
      </label>
      <div style={{ borderTop: "1px solid var(--color-hairline)", paddingTop: "0.9rem" }}>
        <p style={eyebrowStyle}>{fischerColor === "b" ? "Defensive distribution" : "Opening distribution"}</p>
        {isLoading ? <p style={captionStyle}>Loading policy distribution…</p> : (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.25rem" }}>
            {charts.map((chart) => <OpeningPie key={chart.title} title={chart.title} data={chart.data} />)}
          </div>
        )}
        {openingStats && <>
          <p style={{ ...captionStyle, marginTop: "0.5rem" }}>{openingStats.sample_size} Fischer {fischerColor === "b" ? "Black defensive" : "White opening"} positions</p>
          <p style={{ ...captionStyle, marginTop: "0.2rem" }}>{openingStats.source}</p>
        </>}
      </div>
      <HeldoutExamples examples={heldoutExamples?.games ?? []} />
      {error && <p style={{ ...captionStyle, color: "var(--color-danger)", margin: 0 }}>{error}</p>}
    </GlassPanel>
  );
}

function StatCard({ label, value }) {
  return <div style={{ border: "1px solid rgba(201,162,39,0.18)", background: "rgba(255,255,255,0.035)", borderRadius: "var(--radius-sm)", padding: "0.9rem" }}>
    <span style={{ ...captionStyle, display: "block", marginBottom: "0.2rem" }}>{label}</span>
    <strong style={{ color: "var(--color-brass-bright)", fontFamily: "var(--font-mono)", fontSize: "1.3rem" }}>{value}</strong>
  </div>;
}

function OpeningPie({ title, data }) {
  return <div>
    <p style={{ ...captionStyle, textAlign: "center", margin: "0 0 -0.45rem" }}>{title}</p>
    {data.length === 0 ? <p style={{ ...captionStyle, textAlign: "center", padding: "2rem 0" }}>No data</p> : (
      <ResponsiveContainer width="100%" height={165}>
        <PieChart>
          <Tooltip
            formatter={(value) => `${Number(value).toFixed(2)}%`}
            contentStyle={{ background: "#171a20", border: "1px solid rgba(201,162,39,0.35)", borderRadius: 8, color: "#ffffff" }}
            itemStyle={{ color: "#ffffff" }}
            labelStyle={{ color: "#ffffff" }}
          />
          <Pie data={data} dataKey="value" nameKey="move" cx="50%" cy="50%" innerRadius={33} outerRadius={60} paddingAngle={2}>
            {data.map((entry, index) => <Cell key={entry.move} fill={CHART_COLORS[index % CHART_COLORS.length]} />)}
          </Pie>
        </PieChart>
      </ResponsiveContainer>
    )}
    {data.slice(0, 3).map((entry, index) => <div key={entry.move} style={{ display: "flex", justifyContent: "space-between", gap: "0.25rem", fontFamily: "var(--font-mono)", fontSize: "0.72rem", color: "var(--color-text-muted)", marginTop: "0.18rem" }}>
      <span style={{ color: CHART_COLORS[index % CHART_COLORS.length] }}>{entry.move}</span><span>{entry.value.toFixed(1)}%</span>
    </div>)}
  </div>;
}

function HeldoutExamples({ examples }) {
  return <div style={{ borderTop: "1px solid var(--color-hairline)", paddingTop: "0.9rem" }}>
    <p style={eyebrowStyle}>Strict hold-out examples</p>
    {examples.length === 0 ? <p style={captionStyle}>Loading Fischer/AI comparisons…</p> : examples.map((game) => (
      <div key={game.game_id} style={{ marginTop: "0.65rem" }}>
        <p style={{ ...captionStyle, color: "var(--color-brass-bright)", marginBottom: "0.3rem" }}>Game {game.game_id}</p>
        {game.positions.map((position) => (
          <div key={position.fen} style={{ borderLeft: "2px solid var(--color-hairline-strong)", marginBottom: "0.35rem", paddingLeft: "0.5rem" }}>
            <p style={captionStyle}>Move {position.fullmove_number} · Fischer {position.fischer_color}</p>
            <p style={{ ...captionStyle, color: "var(--color-text-primary)" }}>Fischer: {position.actual_move.san} · AI: {position.ai_top_moves[0].san}</p>
            <p style={captionStyle}>Top-3: {position.ai_top_moves.map((move) => move.san).join(", ")} · {position.top1_match ? "Top-1 match" : position.top3_match ? "Top-3 match" : "No match"}</p>
          </div>
        ))}
      </div>
    ))}
  </div>;
}

const eyebrowStyle = { fontFamily: "var(--font-mono)", fontSize: "0.78rem", letterSpacing: "0.12em", textTransform: "uppercase", color: "var(--color-brass)", margin: "0 0 0.35rem" };
const captionStyle = { fontFamily: "var(--font-mono)", fontSize: "0.76rem", letterSpacing: "0.02em", color: "var(--color-text-muted)", margin: 0 };
const uploadLabelStyle = { display: "block", border: "1px dashed var(--color-hairline-strong)", borderRadius: "var(--radius-sm)", color: "var(--color-brass-bright)", cursor: "pointer", fontFamily: "var(--font-mono)", fontSize: "0.82rem", letterSpacing: "0.04em", padding: "0.7rem", textAlign: "center" };
const heatmapLabelStyle = { display: "flex", alignItems: "center", gap: "0.55rem", color: "var(--color-text-muted)", cursor: "pointer", fontFamily: "var(--font-mono)", fontSize: "0.76rem" };
