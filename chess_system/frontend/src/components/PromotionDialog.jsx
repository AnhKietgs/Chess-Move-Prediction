import React from "react";

/**
 * Overlay letting the player pick which piece a pawn promotes to.
 * Fully custom (not react-chessboard's built-in dialog) so it works
 * identically whether the promoting move came from drag-and-drop or
 * click-to-move, and so it matches the glass/brass design system.
 *
 * @param {{
 *   color: "w"|"b",
 *   onSelect: (piece: "q"|"r"|"b"|"n") => void,
 *   onCancel: () => void,
 * }} props
 */
export default function PromotionDialog({ color, onSelect, onCancel }) {
  const options = [
    { piece: "q", label: "Hậu", glyph: color === "w" ? "♕" : "♛" },
    { piece: "r", label: "Xe", glyph: color === "w" ? "♖" : "♜" },
    { piece: "b", label: "Tượng", glyph: color === "w" ? "♗" : "♝" },
    { piece: "n", label: "Mã", glyph: color === "w" ? "♘" : "♞" },
  ];

  return (
    <div
      onClick={onCancel}
      style={{
        position: "absolute",
        inset: 0,
        zIndex: 10,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "rgba(11, 13, 18, 0.55)",
        backdropFilter: "blur(2px)",
        borderRadius: "inherit",
      }}
    >
      <div
        onClick={(event) => event.stopPropagation()}
        style={{
          background: "var(--color-glass-strong)",
          backdropFilter: "blur(18px)",
          WebkitBackdropFilter: "blur(18px)",
          border: "1px solid var(--color-hairline-strong)",
          borderRadius: "var(--radius-md)",
          boxShadow: "var(--shadow-panel)",
          padding: "1.25rem",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: "0.9rem",
        }}
      >
        <p
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.7rem",
            letterSpacing: "0.14em",
            textTransform: "uppercase",
            color: "var(--color-brass)",
            margin: 0,
          }}
        >
          Phong cấp thành
        </p>
        <div style={{ display: "flex", gap: "0.6rem" }}>
          {options.map(({ piece, label, glyph }) => (
            <button
              key={piece}
              onClick={() => onSelect(piece)}
              title={label}
              style={{
                width: 64,
                height: 64,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                gap: "0.15rem",
                background: color === "w" ? "var(--color-ivory)" : "var(--color-walnut)",
                border: "1px solid var(--color-hairline)",
                borderRadius: "var(--radius-sm)",
                color: color === "w" ? "var(--color-walnut)" : "var(--color-ivory)",
                transition: "transform 0.12s ease, border-color 0.12s ease",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = "translateY(-3px)";
                e.currentTarget.style.borderColor = "var(--color-brass-bright)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = "translateY(0)";
                e.currentTarget.style.borderColor = "var(--color-hairline)";
              }}
            >
              <span style={{ fontSize: "2rem", lineHeight: 1 }}>{glyph}</span>
              <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.62rem" }}>{label}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
