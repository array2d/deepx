interface Props {
  onMouseDown: (e: React.MouseEvent) => void;
  active: boolean;
}

export function ResizeHandle({ onMouseDown, active }: Props) {
  return (
    <div
      className={`resize-handle ${active ? 'active' : ''}`}
      onMouseDown={onMouseDown}
    />
  );
}
