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
    displayGame,
    displayFen,
    displayLastMove,
    visibleHistory,
    displayedPly,
    isReviewingHistory,
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
    <div className="play-page">
      {phase === "color-select" && <ColorSelectModal onChoose={chooseColor} />}

      <header className="play-page__header">
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
        <h1 className="play-page__title">Chess AI</h1>
      </header>

      {phase === "playing" && (
        <MainPlayArea
          game={displayGame}
          fen={displayFen}
          playerColor={playerColor}
          history={history}
          visibleHistory={visibleHistory}
          lastMove={displayLastMove}
          displayedPly={displayedPly}
          isReviewingHistory={isReviewingHistory}
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
