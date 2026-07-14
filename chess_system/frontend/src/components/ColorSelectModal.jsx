import React from "react";
import GlassPanel from "./GlassPanel.jsx";

/**
 * Full-screen overlay asking the player which side they want to play
 * before the board is shown. This is the "signature" moment of the
 * landing experience — two brass-edged glass tiles like study lamps.
 *
 * @param {{onChoose: (color: "w"|"b") => void}} props
 */
export default function ColorSelectModal({ onChoose }) {
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        flexDirection: "column",
        gap: "2.5rem",
        padding: "2rem",
        zIndex: 50,
      }}
    >
      <div style={{ textAlign: "center", maxWidth: 520 }}>
        <p
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.75rem",
            letterSpacing: "0.18em",
            textTransform: "uppercase",
            color: "var(--color-brass)",
            margin: "0 0 0.75rem",
          }}
        >
          Behavioral Cloning · Bobby Fischer
        </p>
        <h1 style={{ fontSize: "clamp(1.8rem, 4vw, 2.6rem)", color: "var(--color-text-primary)" }}>
          Take a seat across the board.
        </h1>
        <p style={{ color: "var(--color-text-muted)", marginTop: "0.75rem", lineHeight: 1.6 }}>
          Choose your color. The AI plays in Fischer's style — sharp,
          principled, and unwilling to give you anything for free.
        </p>
      </div>

      <div style={{ display: "flex", gap: "1.5rem", flexWrap: "wrap", justifyContent: "center" }}>
        <ColorTile
          color="w"
          label="Play White"
          sublabel="You move first"
          onChoose={onChoose}
        />
        <ColorTile
          color="b"
          label="Play Black"
          sublabel="Fischer opens"
          onChoose={onChoose}
        />
      </div>
    </div>
  );
}

function ColorTile({ color, label, sublabel, onChoose }) {
  const isWhite = color === "w";
  return (
    <GlassPanel
      as="button"
      className="color-tile"
      style={{
        width: 220,
        padding: "2rem 1.5rem",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "0.85rem",
        color: "var(--color-text-primary)",
        cursor: "pointer",
        transition: "transform 0.15s ease, border-color 0.15s ease",
      }}
    >
      <button
        onClick={() => onChoose(color)}
        style={{
          all: "unset",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: "0.85rem",
          width: "100%",
        }}
      >
        <span
          aria-hidden
          style={{
            width: 56,
            height: 56,
            borderRadius: "50%",
            background: isWhite ? "var(--color-ivory)" : "var(--color-walnut)",
            border: `1px solid ${isWhite ? "var(--color-hairline-strong)" : "var(--color-hairline)"}`,
            boxShadow: "inset 0 0 12px rgba(0,0,0,0.25)",
          }}
        />
        <span style={{ fontFamily: "var(--font-display)", fontSize: "1.15rem" }}>{label}</span>
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.72rem",
            color: "var(--color-text-muted)",
            textTransform: "uppercase",
            letterSpacing: "0.08em",
          }}
        >
          {sublabel}
        </span>
      </button>
    </GlassPanel>
  );
}
