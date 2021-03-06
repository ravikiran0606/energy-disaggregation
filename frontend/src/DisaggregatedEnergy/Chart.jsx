import React, { useState } from 'react';
import { VictoryChart, VictoryLine, VictoryZoomContainer, VictoryBrushContainer, VictoryAxis, VictoryLegend } from 'victory';

const Chart = ({ values, xAxis, mainsOne, mainsTwo, appliance }) => {
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
        <VictoryLegend x={400} y={50}
          orientation="horizontal"
          gutter={20}
          style={{ border: { stroke: "black" }, title: {fontSize: 20 } }}
          data={[
            { name: `${appliance.charAt(0).toUpperCase()}${appliance.substring(1)}`, symbol: { fill: "#BB4430" } },
            { name: "Mains 1", symbol: { fill: "#002642" } },
            { name: "Mains 2", symbol: { fill: "#248232" } }
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
          data={values}
        />
        <VictoryLine
          style={{
            data: {stroke: "#002642"}
          }}
          data={mainsOne}
        />
        <VictoryLine
          style={{
            data: {stroke: "#248232"}
          }}
          data={mainsTwo}
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
          data={values}
        />
        <VictoryLine
          style={{
            data: {stroke: "#002642"}
          }}
          data={mainsOne}
        />
        <VictoryLine
          style={{
            data: {stroke: "#248232"}
          }}
          data={mainsTwo}
        />
      </VictoryChart>
    </div>
  );
};

export default Chart;
