import React from "react";

/**
 * The base glass surface used across the app: blurred translucent fill,
 * a hairline brass border, and a soft drop shadow. Every panel-like
 * surface (board frame, sidebar, modal) should compose this rather than
 * redefine the glass effect inline.
 *
 * @param {{children: React.ReactNode, className?: string, style?: React.CSSProperties, as?: keyof JSX.IntrinsicElements}} props
 */
export default function GlassPanel({ children, className = "", style = {}, as: Tag = "div" }) {
  return (
    <Tag
      className={`glass-panel ${className}`}
      style={{
        background: "var(--color-glass)",
        backdropFilter: "blur(18px)",
        WebkitBackdropFilter: "blur(18px)",
        border: "1px solid var(--color-hairline)",
        borderRadius: "var(--radius-lg)",
        boxShadow: "var(--shadow-panel)",
        ...style,
      }}
    >
      {children}
    </Tag>
  );
}
