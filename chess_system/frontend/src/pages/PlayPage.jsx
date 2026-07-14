import React from "react";
import { useGameContext } from "../context/GameContext.jsx";
import { useChessGame } from "../hooks/useChessGame.js";
import ColorSelectModal from "../components/ColorSelectModal.jsx";
import ChessBoardContainer from "../components/ChessBoardContainer.jsx";
import MoveHistory from "../components/MoveHistory.jsx";
import GlassPanel from "../components/GlassPanel.jsx";

export default function PlayPage() {
  const { playerColor, phase, chooseColor, resetToColorSelect } = useGameContext();
  const {
    game,
    fen,
    history,
    lastMove,
    isAiThinking,
    statusMessage,
    errorMessage,
    makePlayerMove,
    resetGame,
    isPlayerTurn,
  } = useChessGame(playerColor);

  const handleNewGame = () => {
    resetGame();
    resetToColorSelect();
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        padding: "2.5rem 1.5rem",
      }}
    >
      {phase === "color-select" && <ColorSelectModal onChoose={chooseColor} />}

      <header style={{ textAlign: "center", marginBottom: "2rem" }}>
        <p
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.72rem",
            letterSpacing: "0.18em",
            textTransform: "uppercase",
            color: "var(--color-brass)",
            margin: 0,
          }}
        >
          Fischer Study
        </p>
        <h1 style={{ fontSize: "1.6rem" }}>Style-Constrained Chess AI</h1>
      </header>

      {phase === "playing" && (
        <div
          style={{
            display: "flex",
            gap: "1.5rem",
            alignItems: "flex-start",
            flexWrap: "wrap",
            justifyContent: "center",
          }}
        >
          <ChessBoardContainer
            game={game}
            fen={fen}
            playerColor={playerColor}
            lastMove={lastMove}
            isLocked={!isPlayerTurn}
            onMove={makePlayerMove}
          />

          <div style={{ display: "flex", flexDirection: "column", gap: "1rem", height: 480 }}>
            <StatusBar
              isAiThinking={isAiThinking}
              statusMessage={statusMessage}
              errorMessage={errorMessage}
              onNewGame={handleNewGame}
            />
            <MoveHistory history={history} />
          </div>
        </div>
      )}
    </div>
  );
}

function StatusBar({ isAiThinking, statusMessage, errorMessage, onNewGame }) {
  return (
    <GlassPanel style={{ padding: "1rem 1.25rem", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
      <span
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: "0.8rem",
          color: errorMessage ? "var(--color-danger)" : "var(--color-text-primary)",
        }}
      >
        {errorMessage
          ? errorMessage
          : isAiThinking
          ? "Fischer is thinking…"
          : statusMessage || "Your move."}
      </span>
      <button
        onClick={onNewGame}
        style={{
          alignSelf: "flex-start",
          background: "transparent",
          border: "1px solid var(--color-hairline-strong)",
          color: "var(--color-brass-bright)",
          borderRadius: "var(--radius-sm)",
          padding: "0.4rem 0.9rem",
          fontFamily: "var(--font-mono)",
          fontSize: "0.72rem",
          letterSpacing: "0.05em",
          textTransform: "uppercase",
        }}
      >
        New Game
      </button>
    </GlassPanel>
  );
}
