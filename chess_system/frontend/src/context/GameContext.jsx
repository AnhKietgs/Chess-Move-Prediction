import React, { createContext, useContext, useMemo, useState } from "react";

/**
 * @typedef {"color-select" | "playing"} GamePhase
 * @typedef {"w" | "b"} PlayerColor
 */

const GameContext = createContext(null);

/**
 * Holds cross-cutting game setup state (which color the human plays, and
 * whether we're still in the pre-game color-select screen). Board state
 * itself (position, move history) lives in `useChessGame`, kept separate
 * so this context stays small and stable.
 */
export function GameProvider({ children }) {
  /** @type {[PlayerColor|null, Function]} */
  const [playerColor, setPlayerColor] = useState(null);
  /** @type {[GamePhase, Function]} */
  const [phase, setPhase] = useState("color-select");

  const chooseColor = (color) => {
    setPlayerColor(color);
    setPhase("playing");
  };

  const resetToColorSelect = () => {
    setPlayerColor(null);
    setPhase("color-select");
  };

  const value = useMemo(
    () => ({ playerColor, phase, chooseColor, resetToColorSelect }),
    [playerColor, phase]
  );

  return <GameContext.Provider value={value}>{children}</GameContext.Provider>;
}

export function useGameContext() {
  const ctx = useContext(GameContext);
  if (!ctx) {
    throw new Error("useGameContext must be used within a GameProvider");
  }
  return ctx;
}
