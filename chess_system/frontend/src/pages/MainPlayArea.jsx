import React, { useState } from "react";
import ChessBoardContainer from "../components/ChessBoardContainer.jsx";
import Dashboard from "../components/Dashboard.jsx";
import GlassPanel from "../components/GlassPanel.jsx";
import MaterialBalance from "../components/MaterialBalance.jsx";
import MoveHistory from "../components/MoveHistory.jsx";

/**
 * Arrange the active board, game controls, scoresheet, and style dashboard.
 *
 * @param {{game: import("chess.js").Chess, fen: string, playerColor: "w"|"b", history: object[], lastMove: {from: string, to: string}|null, isAiThinking: boolean, statusMessage: string, errorMessage: string, isPlayerTurn: boolean, onMove: Function, onNewGame: Function}} props
 * @returns {React.JSX.Element} Main game layout.
 */
export default function MainPlayArea({
  game,
  fen,
  playerColor,
  history,
  lastMove,
  isAiThinking,
  statusMessage,
  errorMessage,
  isPlayerTurn,
  onMove,
  onNewGame,
  onResign,
  hasResigned,
}) {
  const [showHeatmap, setShowHeatmap] = useState(false);
  const aiMoveSquares = history
    .filter((move) => move.color !== playerColor)
    .map((move) => move.to);

  return (
    <div style={{ display: "flex", gap: "1.5rem", alignItems: "stretch", flexWrap: "wrap", justifyContent: "center" }}>
      <div style={{ alignSelf: "flex-start" }}>
        <Dashboard
          showHeatmap={showHeatmap}
          onHeatmapChange={setShowHeatmap}
          fischerColor={playerColor === "w" ? "b" : "w"}
        />
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: "0.7rem", alignSelf: "flex-start" }}>
        <MaterialBalance history={history} playerColor={playerColor} position="top" />
        <ChessBoardContainer
          game={game}
          fen={fen}
          playerColor={playerColor}
          lastMove={lastMove}
          isLocked={!isPlayerTurn}
          aiMoveSquares={showHeatmap ? aiMoveSquares : []}
          onMove={onMove}
        />
        <MaterialBalance history={history} playerColor={playerColor} position="bottom" />
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: "1rem", width: 320, height: "calc(min(84vw, 740px) + 176px)", minHeight: 0 }}>
        <StatusBar
          isAiThinking={isAiThinking}
          statusMessage={statusMessage}
          errorMessage={errorMessage}
          onNewGame={onNewGame}
          onResign={onResign}
          hasResigned={hasResigned}
        />
        <MoveHistory history={history} />
      </div>
    </div>
  );
}

function StatusBar({ isAiThinking, statusMessage, errorMessage, onNewGame, onResign, hasResigned }) {
  return (
    <GlassPanel style={{ padding: "1rem 1.25rem", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
      <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.8rem", color: errorMessage ? "var(--color-danger)" : "var(--color-text-primary)" }}>
        {errorMessage ? errorMessage : isAiThinking ? "Fischer is thinking…" : statusMessage || "Your move."}
      </span>
      <div style={{ display: "flex", gap: "0.55rem", flexWrap: "wrap" }}>
        <button onClick={onNewGame} style={newGameButtonStyle}>New Game</button>
        <button onClick={onResign} disabled={hasResigned} style={{ ...resignButtonStyle, opacity: hasResigned ? 0.45 : 1 }}>Resign</button>
      </div>
    </GlassPanel>
  );
}

const newGameButtonStyle = { alignSelf: "flex-start", background: "transparent", border: "1px solid var(--color-hairline-strong)", color: "var(--color-brass-bright)", borderRadius: "var(--radius-sm)", padding: "0.4rem 0.9rem", fontFamily: "var(--font-mono)", fontSize: "0.72rem", letterSpacing: "0.05em", textTransform: "uppercase" };
const resignButtonStyle = { ...newGameButtonStyle, borderColor: "rgba(179,84,63,0.7)", color: "#d98270" };
