import React, { useState } from "react";
import ChessBoardContainer from "../components/ChessBoardContainer.jsx";
import Dashboard from "../components/Dashboard.jsx";
import GlassPanel from "../components/GlassPanel.jsx";
import MaterialBalance from "../components/MaterialBalance.jsx";
import MoveHistory from "../components/MoveHistory.jsx";
import fischerPortrait from "../components/bobby-fischer.jpg";

/**
 * Arrange the active board, game controls, scoresheet, and style dashboard.
 *
 * @param {{game: import("chess.js").Chess, fen: string, playerColor: "w"|"b", history: object[], visibleHistory: object[], lastMove: {from: string, to: string}|null, displayedPly: number, isReviewingHistory: boolean, isAiThinking: boolean, statusMessage: string, errorMessage: string, isPlayerTurn: boolean, onMove: Function, onNewGame: Function}} props
 * @returns {React.JSX.Element} Main game layout.
 */
export default function MainPlayArea({
  game,
  fen,
  playerColor,
  history,
  visibleHistory,
  lastMove,
  displayedPly,
  isReviewingHistory,
  isAiThinking,
  statusMessage,
  errorMessage,
  isPlayerTurn,
  onMove,
  onNewGame,
  onResign,
  hasResigned,
  useSafetyNet,
  onSafetyNetChange,
}) {
  const [showHeatmap, setShowHeatmap] = useState(false);
  const aiMoveSquares = visibleHistory
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
        <MaterialBalance history={visibleHistory} playerColor={playerColor} position="top" />
        <ChessBoardContainer
          game={game}
          fen={fen}
          playerColor={playerColor}
          lastMove={lastMove}
          isLocked={!isPlayerTurn || isReviewingHistory}
          aiMoveSquares={showHeatmap ? aiMoveSquares : []}
          onMove={onMove}
        />
        <MaterialBalance history={visibleHistory} playerColor={playerColor} position="bottom" />
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: "1rem", width: 320, height: "calc(min(84vw, 740px) + 176px)", minHeight: 0 }}>
        <StatusBar
          isAiThinking={isAiThinking}
          isReviewingHistory={isReviewingHistory}
          displayedPly={displayedPly}
          totalPly={history.length}
          statusMessage={statusMessage}
          errorMessage={errorMessage}
          onNewGame={onNewGame}
          onResign={onResign}
          hasResigned={hasResigned}
          useSafetyNet={useSafetyNet}
          onSafetyNetChange={onSafetyNetChange}
        />
        <MoveHistory history={history} displayedPly={displayedPly} />
      </div>
    </div>
  );
}

function StatusBar({ isAiThinking, isReviewingHistory, displayedPly, totalPly, statusMessage, errorMessage, onNewGame, onResign, hasResigned, useSafetyNet, onSafetyNetChange }) {
  const reviewMessage = "Viewing move " + displayedPly + "/" + totalPly + " — use ← →";
  const message = errorMessage || (isAiThinking ? "Fischer is thinking…" : isReviewingHistory ? reviewMessage : statusMessage || "Your move.");

  return (
    <GlassPanel style={{ padding: "1rem 1.25rem", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "0.55rem", minHeight: "2rem" }}>
        {isAiThinking && <img
          src={fischerPortrait}
          alt="Bobby Fischer"
          title="Bobby Fischer"
          style={thinkingAvatarStyle}
        />}
        <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.8rem", color: errorMessage ? "var(--color-danger)" : "var(--color-text-primary)" }}>
          {message}
        </span>
      </div>
      <div style={{ display: "flex", gap: "0.55rem", flexWrap: "wrap" }}>
        <button onClick={onNewGame} style={newGameButtonStyle}>New Game</button>
        <button onClick={onResign} disabled={hasResigned} style={{ ...resignButtonStyle, opacity: hasResigned ? 0.45 : 1 }}>Resign</button>
      </div>
      <label style={safetyNetToggleStyle}>
        <input
          type="checkbox"
          checked={useSafetyNet}
          onChange={(event) => onSafetyNetChange(event.target.checked)}
        />
        <span>Stockfish safety-net</span>
      </label>
    </GlassPanel>
  );
}

const newGameButtonStyle = { alignSelf: "flex-start", background: "transparent", border: "1px solid var(--color-hairline-strong)", color: "var(--color-brass-bright)", borderRadius: "var(--radius-sm)", padding: "0.4rem 0.9rem", fontFamily: "var(--font-mono)", fontSize: "0.72rem", letterSpacing: "0.05em", textTransform: "uppercase" };
const resignButtonStyle = { ...newGameButtonStyle, borderColor: "rgba(179,84,63,0.7)", color: "#d98270" };
const safetyNetToggleStyle = { display: "flex", alignItems: "center", gap: "0.45rem", color: "var(--color-text-muted)", cursor: "pointer", fontFamily: "var(--font-mono)", fontSize: "0.68rem" };
const thinkingAvatarStyle = { width: "2rem", height: "2rem", border: "1px solid var(--color-brass)", borderRadius: "50%", flex: "0 0 auto", objectFit: "cover" };
