import React from "react";
import { useGameContext } from "../context/GameContext.jsx";
import { useChessGame } from "../hooks/useChessGame.js";
import ColorSelectModal from "../components/ColorSelectModal.jsx";
import MainPlayArea from "./MainPlayArea.jsx";

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
    resignGame,
    hasResigned,
    useSafetyNet,
    setUseSafetyNet,
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
        <h1 style={{ fontSize: "2rem" }}>Style-Constrained Chess AI</h1>
      </header>

      {phase === "playing" && (
        <MainPlayArea
          game={game}
          fen={fen}
          playerColor={playerColor}
          history={history}
          lastMove={lastMove}
          isAiThinking={isAiThinking}
          statusMessage={statusMessage}
          errorMessage={errorMessage}
          isPlayerTurn={isPlayerTurn}
          onMove={makePlayerMove}
          onNewGame={handleNewGame}
          onResign={resignGame}
          hasResigned={hasResigned}
          useSafetyNet={useSafetyNet}
          onSafetyNetChange={setUseSafetyNet}
        />
      )}
    </div>
  );
}
