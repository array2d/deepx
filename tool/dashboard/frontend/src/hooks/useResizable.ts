import { useState, useCallback, useEffect, useRef } from 'react';

interface UseResizableOptions {
  initialSize: number;
  minSize: number;
  maxSize: number;
  direction: 'left' | 'right';
}

export function useResizable({ initialSize, minSize, maxSize, direction }: UseResizableOptions) {
  const [size, setSize] = useState(initialSize);
  const [dragging, setDragging] = useState(false);
  const startPosRef = useRef(0);
  const startSizeRef = useRef(initialSize);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setDragging(true);
    startPosRef.current = e.clientX;
    startSizeRef.current = size;
  }, [size]);

  useEffect(() => {
    if (!dragging) return;

    const onMouseMove = (e: MouseEvent) => {
      const delta = e.clientX - startPosRef.current;
      const newSize = direction === 'left'
        ? startSizeRef.current + delta
        : startSizeRef.current - delta;
      setSize(Math.max(minSize, Math.min(maxSize, newSize)));
    };

    const onMouseUp = () => {
      setDragging(false);
    };

    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
    return () => {
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };
  }, [dragging, direction, minSize, maxSize]);

  return { size, dragging, onMouseDown };
}
