import React, { useState } from 'react';
import { VictoryChart, VictoryLine, VictoryZoomContainer, VictoryBrushContainer, VictoryAxis } from 'victory';

const Chart = ({ values, xAxis, yAxis }) => {
  const [selectedDomain, setSelectedDomain] = useState(null);
  const [zoomDomain, setZoomDomain] = useState(null);

  return (
    <div>
      <VictoryChart
        width={550}
        height={300}
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
        <VictoryAxis
          tickCount={5}
          tickValues={xAxis}
        />
        <VictoryAxis
          dependentAxis
          
        />
        <VictoryLine
          style={{
            data: {stroke: "tomato"}
          }}
          data={values}
        />
      </VictoryChart>
      <VictoryChart
        width={550}
        height={90}
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
        />
        <VictoryLine
          style={{
            data: {stroke: "tomato"}
          }}
          data={values}
        />
      </VictoryChart>
    </div>
  );
};

export default Chart;
