import React, { useState } from 'react';
import { VictoryChart, VictoryLine, VictoryZoomContainer, VictoryBrushContainer, VictoryAxis, VictoryLegend } from 'victory';

const Chart = ({ historical, forecast, xAxis }) => {
  const [selectedDomain, setSelectedDomain] = useState(null);
  const [zoomDomain, setZoomDomain] = useState(null);

  return (
    <div>
      <VictoryChart
        width={750}
        height={500}
        scale={{x: "time"}}
        domainPadding={{y: 25}}
        containerComponent={
          <VictoryZoomContainer responsive={false}
            zoomDimension="x"
            zoomDomain={zoomDomain}
            onZoomDomainChange={setSelectedDomain}
          />
        }
      >
        <VictoryLegend x={500} y={50}
          orientation="horizontal"
          gutter={20}
          style={{ border: { stroke: "black" }, title: {fontSize: 20 } }}
          data={[
            { name: 'Historical', symbol: { fill: "#BB4430" } },
            { name: "Forecast", symbol: { fill: "#002642" } }
          ]}
        />
        <VictoryAxis
          tickCount={5}
          tickValues={xAxis}
          orientation="bottom"
          offsetY={50}
        />
        <VictoryAxis
          dependentAxis
        />
        <VictoryLine
          style={{
            data: {stroke: "#BB4430"}
          }}
          data={historical}
        />
        <VictoryLine
          style={{
            data: {stroke: "#002642"}
          }}
          data={forecast}
        />
      </VictoryChart>
      <VictoryChart
        width={750}
        height={125}
        scale={{x: "time"}}
        domainPadding={{y: 15}}
        padding={{top: 0, left: 50, right: 50, bottom: 30}}
        containerComponent={
          <VictoryBrushContainer responsive={false}
            brushDimension="x"
            brushDomain={selectedDomain}
            onBrushDomainChange={setZoomDomain}
          />
        }
      >
        <VictoryAxis
          tickCount={5}
          tickValues={xAxis}
          orientation="bottom"
          offsetY={30}
        />
        <VictoryLine
          style={{
            data: {stroke: "#BB4430"}
          }}
          data={historical}
        />
        <VictoryLine
          style={{
            data: {stroke: "#002642"}
          }}
          data={forecast}
        />
      </VictoryChart>
    </div>
  );
};

export default Chart;
