import React from "react";
import { GameProvider } from "./context/GameContext.jsx";
import PlayPage from "./pages/PlayPage.jsx";

export default function App() {
  return (
    <GameProvider>
      <PlayPage />
    </GameProvider>
  );
}
