import React from 'react';

type ComparisonTableProps = {
  headers: string[];
  rows: string[][];
};

/**
 * 보험 상품 비교 테이블 컴포넌트.
 */
export function ComparisonTable({ headers, rows }: ComparisonTableProps): JSX.Element {
  return (
    <div className="mt-4 overflow-x-auto">
      <div className="bg-white/80 backdrop-blur-md rounded-2xl p-4 border border-gray-200/50">
        <h4 className="text-gray-800 font-semibold mb-3 text-base">📊 보험 상품 비교</h4>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200/50">
              {headers.map((header, index) => (
                <th
                  key={index}
                  className="text-left py-2 px-3 text-gray-700 font-medium first:pl-0 last:pr-0"
                >
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={rowIndex} className="border-b border-gray-100/50 last:border-b-0">
                {row.map((cell, cellIndex) => (
                  <td
                    key={cellIndex}
                    className="py-2 px-3 text-gray-600 first:pl-0 last:pr-0"
                  >
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
