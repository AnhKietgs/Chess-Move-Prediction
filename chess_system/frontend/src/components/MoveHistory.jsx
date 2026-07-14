import React, { useEffect, useRef } from "react";
import GlassPanel from "./GlassPanel.jsx";

/**
 * Right-hand sidebar rendering standard algebraic notation as a
 * brass-ruled ledger — the signature element of the design: it reads
 * like a tournament scoresheet rather than a chat log.
 *
 * @param {{history: {moveNumber: number, san: string, color: "w"|"b"}[]}} props
 */
export default function MoveHistory({ history }) {
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [history.length]);

  // Group into rows of [white move, black move] keyed by move number.
  const rows = [];
  history.forEach((entry) => {
    const rowIndex = entry.moveNumber - 1;
    if (!rows[rowIndex]) rows[rowIndex] = { moveNumber: entry.moveNumber, w: "", b: "" };
    rows[rowIndex][entry.color] = entry.san;
  });

  return (
    <GlassPanel
      style={{ width: 280, height: "100%", display: "flex", flexDirection: "column", padding: "1.25rem" }}
    >
      <h2 style={{ fontSize: "1rem", color: "var(--color-brass-bright)", marginBottom: "0.25rem" }}>
        Scoresheet
      </h2>
      <p
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: "0.68rem",
          color: "var(--color-text-muted)",
          letterSpacing: "0.06em",
          margin: "0 0 1rem",
        }}
      >
        {history.length === 0 ? "No moves yet" : `${history.length} ply recorded`}
      </p>

      <div
        ref={scrollRef}
        style={{
          flex: 1,
          overflowY: "auto",
          borderTop: "1px solid var(--color-hairline)",
        }}
      >
        {rows.map((row) => (
          <div
            key={row.moveNumber}
            style={{
              display: "grid",
              gridTemplateColumns: "2rem 1fr 1fr",
              gap: "0.5rem",
              padding: "0.4rem 0",
              borderBottom: "1px solid rgba(201,162,39,0.08)",
              fontFamily: "var(--font-mono)",
              fontSize: "0.85rem",
            }}
          >
            <span style={{ color: "var(--color-text-muted)" }}>{row.moveNumber}.</span>
            <span style={{ color: "var(--color-ivory)" }}>{row.w}</span>
            <span style={{ color: "var(--color-slate)" }}>{row.b}</span>
          </div>
        ))}
      </div>
    </GlassPanel>
  );
}
