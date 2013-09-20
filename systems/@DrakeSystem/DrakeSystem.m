classdef DrakeSystem < DynamicalSystem
% A DynamicalSystem with the functionality (dynamics, update, outputs, 
% etc) implemented in matlab, so that it is amenable to, for instance, symbolic
% manipulations.  These functions are wrapped as an S-Function in
% DCSFunction.cpp.

  % constructor
  methods
    function obj = DrakeSystem(num_xc,num_xd,num_u,num_y,direct_feedthrough_flag,time_invariant_flag)
      % Construct a DrakeSystem
      %
      % @param num_xc number of continuous-time state variables
      % @param num_xd number of discrete-time state variables
      % @param num_u number of inputs
      % @param num_y number of outputs
      % @param direct_feedthrough_flag true means that the output depends
      %   directly on the input.  Set to false if possible.
      % @param time_invariant_flag true means that the
      %   dynamics/update/output do not depend on time.  Set to true if
      %   possible.
      
      obj.uid = sprintf('%018.0f', now * 24*60*60*1e6);

      if (nargin>0)
        obj = setNumContStates(obj,num_xc);
        obj = setNumDiscStates(obj,num_xd);
        obj = setNumInputs(obj,num_u);
        if (nargin>=4), obj = setNumOutputs(obj,num_y); else obj = setNumOutputs(obj,0); end
        if (nargin>=5), obj = setDirectFeedthrough(obj,direct_feedthrough_flag); end
        if (nargin>=6), obj = setTIFlag(obj,time_invariant_flag); end
      end
      obj = setParamFrame(obj,CoordinateFrame([class(obj),'Params'],0,'p'));  % no parameters by default
    end      
  end
  
  % default methods - these should be implemented or overwritten
  % 
  methods
    function x0 = getInitialState(obj)
      % Return a (potentially random) state double (column) vector of initial conditions
      %
      % This method is intended to be overloaded, but by default attempts to
      % return the result of resolveConstraints using the zero vector as an
      % initial seed.
      x0 = zeros(obj.num_xd+obj.num_xc,1);
      attempts=0;
      success=false;
      tries = 0;
      while (~success)
        try
          [x0,success] = resolveConstraints(obj,x0);
        catch ex
          if strcmp(ex.identifier,'Drake:DrakeSystem:FailedToResolveConstraints');
            attempts = attempts+1;
            if (attempts>=10)
              error('Drake:Manipulator:FailedToResolveConstraints','Failed to resolve state constraints on initial conditions after 10 tries');
            else
              x0 = randn(obj.num_xd+obj.num_xc,1);
              continue;
            end
          else
            rethrow(ex);
          end
        end
        if (~success)
          x0 = randn(obj.num_xd+obj.num_xc,1);
          tries = tries+1;
          if (tries>=10)
              error('failed to resolve constraints after %d attempts',tries);
          end
        end
      end
      x0 = double(x0);
    end
    
    function xcdot = dynamics(obj,t,x,u)
      % Placeholder for the dynamics method.  Systems with continuous state
      % must overload this method.
      error('Drake:DrakeSystem:AbstractMethod','systems with continuous states must implement Derivatives (ie overload dynamics function)');
    end
    
    function xdn = update(obj,t,x,u)
      % Placeholder for the update method.  Systems with discrete state
      % must overload this method.
      error('Drake:DrakeSystem:AbstractMethod','systems with discrete states must implement Update (ie overload update function)');
    end
    
    function y = output(obj,t,x,u)
      % Placeholder for the output method.  Systems must overload this method.
      error('Drake:DrakeSystem:AbstractMethod','default is intentionally not implemented');
    end
    
    function zcs = zeroCrossings(obj,t,x,u)
      % Placeholder for the zeroCrossings method: a method 
      % phi = zeroCrossings(t,x,u) which triggers a zero crossing 
      % event when phi transitions from positive to negative.  
      %
      % Systems with zero crossings must overload this method.
      error('Drake:DrakeSystem:AbstractMethod','systems with zero crossings must implement the zeroCrossings method'); 
    end
    
  end
  
  % access methods
  methods
    function n = getNumContStates(obj)
      % Returns the number of continuous states
      n = obj.num_xc;
    end
    function n = getNumDiscStates(obj)
      % Returns the number of discrete states
      n = obj.num_xd;
    end
    function n = getNumInputs(obj)
      % Returns the number of inputs to the system
      n = obj.num_u;
    end
    function n = getNumOutputs(obj)
      % Returns the number of outputs from the system
      n = obj.num_y;
    end
    function x0 = getInitialStateWInput(obj,t,x,u)  
      % Hook in case a system needs to initial state based on current time and/or input.  
      % This gets called after getInitialState(), and unfortunately will override inputs supplied by simset.
      x0=x;  % by default, do nothing. 
    end
    function ts = getSampleTime(obj)  
      % As described at http://www.mathworks.com/help/toolbox/simulink/sfg/f6-58760.html
      % to set multiple sample times, specify one *column* for each sample
      % time/offset pair.
      % The default behavior is continuous time for systems with only continuous
      % states, and discrete time (with sample period 1s) for systems only
      % discrete states, and inherited for systems with no states.  For
      % systems with both discrete and continuous states, an error is
      % thrown saying that this function should be overloaded to set the
      % desired behavior.  

      if ~isempty(obj.ts)
        ts = obj.ts;
      elseif (obj.num_xc>0 && obj.num_xd==0)
        ts = [0;0];  % continuous time, no offset
      elseif (obj.num_xc==0 && obj.num_xd>0)
        ts = [1;0];  % discrete time, with period 1s.
      elseif (obj.num_xc==0 && obj.num_xd==0)
        ts = [-1;0]; % inherited sample time
      else
        error('Drake:DrakeSystem:NotImplemented','systems with both discrete and continuous states must implement the getSampleTime method or call setSampleTime to specify the desired behavior');
      end
    end
    
  end
  
  methods (Sealed = true)
    function ts = getInputSampleTimes(obj)
      % Returns getSampleTime - a DrakeSystem can only have a single same
      % time associated with it.
      ts = getSampleTime(obj);
    end
    function ts = getOutputSampleTimes(obj)
      % Returns getSampleTime - a DrakeSystem can only have a single same
      % time associated with it.
      ts = getSampleTime(obj);
    end
  end
  
  methods
    function obj = setSampleTime(obj,ts)
      % robust method for setting default sample time
      % 
      % @param ts a 2-by-n matrix with each column containing a sample time
      %    redundant colums are eliminated automatically.  
      ts = unique(ts','rows')';

      % only a few possibilities are allowed/supported
      %   inherited, single continuous, single discrete, single continuous+single
      %   discrete (note: disabled single continuous + single discrete
      %   because it wasn't obviously the right thing... e.g. in the
      %   visualizer who asked for the output to be at fixed dt, but after
      %   combination, the output gets called continuously).
      if size(ts,2)>1  % if only one ts, then all is well
        if any(ts(1,:)==-1)  % zap superfluous inherited
          ts=ts(:,ts(1,:)~=-1);
        end
        if sum(ts(1,:)>0)>1 % then multiple discrete
          error('Drake:DrakeSystem:UnsupportedSampleTime','cannot define a drakesystem using modes that have different discrete sample times');
        end
        if sum(ts(1,:)==0)>1 % then multiple continuous
          error('Drake:DrakeSystem:UnsupportedSampleTime','cannot define a drakesystem using modes that have both ''continuous time'' and ''continuous time, fixed in minor offset'' sample times');
        end
        if sum(ts(1,:)>=0)>1 % then both continuous and discrete
          error('Drake:DrakeSystem:UnsupportedSampleTime','cannot define a drakesystem using modes that have both continuous and discrete sample times');
        end
      end
      obj.ts = ts;
    end
    function tf = isDirectFeedthrough(obj)
      % Check if the system is direct feedthrough (e.g., if the output
      % depends on the immediate input)
      tf = obj.direct_feedthrough_flag;
    end
    function obj = setDirectFeedthrough(obj,tf)
      % Set the direct feedthrough flag
      obj.direct_feedthrough_flag = tf;
    end
    function mdl = getModel(obj)
      % Constructs a simulink system block for this system to be used by
      % the simulink engine.
      %
      % @retval mdl string id for the simulink system
      
      % First, make sure we have a compiled DCSFunction
      if(~exist('DCSFunction','file'))
        errorMsg={'Sorry, you have not run ''make'' yet in the drake root,'
          'which means you do not have the compiled MEX files needed to run this program.'
          'Running configure and make in the drake root directory will fix this.'};
        error('%s\n',errorMsg{:})
      end
      
      % make a simulink model from this block
      mdl = [class(obj),'_',obj.uid];  % use the class name + uid as the model name
      close_system(mdl,0);  % close it if there is an instance already open
      new_system(mdl,'Model');
      set_param(mdl,'SolverPrmCheckMsg','none');  % disables warning for automatic selection of default timestep
      
      assignin('base',[mdl,'_obj'],obj);
      
      load_system('simulink');
      load_system('simulink3');
      add_block('simulink/User-Defined Functions/S-Function',[mdl,'/DrakeSys'], ...
        'FunctionName','DCSFunction', ...
        'parameters',[mdl,'_obj']);

      m = Simulink.Mask.create([mdl,'/DrakeSys']);
      m.set('Display',['fprintf(''',class(obj),''')']);
      
      if (getNumInputs(obj)>0)
        add_block('simulink3/Sources/In1',[mdl,'/in']);
        
        if (any(~isinf([obj.umin,obj.umax]))) % then add saturation block
          add_block('simulink3/Nonlinear/Saturation',[mdl,'/sat'],...
            'UpperLimit',mat2str(obj.umax),'LowerLimit',mat2str(obj.umin));
          add_line(mdl,'in/1','sat/1');
          add_line(mdl,'sat/1','DrakeSys/1');
        else
          add_line(mdl,'in/1','DrakeSys/1');
        end
      end
      if (getNumOutputs(obj)>0)
        add_block('simulink3/Sinks/Out1',[mdl,'/out']);
        add_line(mdl,'DrakeSys/1','out/1');
      end
      
      if (obj.num_xcon>0)
        warning('Drake:DrakeSystem:ConstraintsNotEnforced','system has constraints, but they aren''t enforced in the simulink model yet.');
      end
    end
    
    function [x,success] = resolveConstraints(obj,x0,v)
      % Attempts to find a x which satisfies the constraints,
      % using x0 as the initial guess.
      %
      % @param x0 initial guess for state satisfying constraints
      % @param v (optional) a visualizer that should be called while the
      % solver is doing it's thing

      if isa(x0,'Point')
        x0 = double(x0.inFrame(obj.getStateFrame));
      end
      
      if (obj.num_xcon < 1)
        x=Point(obj.getStateFrame,x0);
        success=true;
        return;
      end
      
      function stop=drawme(x,optimValues,state)
        stop=false;
        v.draw(0,x);
      end
        
      if (nargin>2 && ~isempty(v))  % useful for debugging (only but only works for URDF manipulators)
        options=optimset('Display','iter','Algorithm','levenberg-marquardt','OutputFcn',@drawme,'TolX',1e-9);
      else
        options=optimset('Display','off','Algorithm','levenberg-marquardt');
      end
      [x,~,exitflag] = fsolve(@(x)stateConstraints(obj,x),x0,options);      
      success=(exitflag==1);
      if (nargout<2 && ~success)
        error('Drake:DrakeSystem:ResolveConstraintsFailed','failed to resolve constraints');
      end
      x = Point(obj.getStateFrame,x);
    end
    
    function [xstar,ustar,success] = findFixedPoint(obj,x0,u0,v)
      % attempts to find a fixed point (xstar,ustar) which also satisfies the constraints,
      % using (x0,u0) as the initial guess.  
      %
      % @param x0 initial guess for the state
      % @param u0 initial guess for the input
      % @param v (optional) a visualizer that should be called while the
      % solver is doing it's thing  
      %
      
      if isa(x0,'Point')
        x0 = double(x0.inFrame(obj.getStateFrame));
      end
      if isa(u0,'Point')
        u0 = double(u0.inFrame(obj.getInputFrame));
      end
   
      if ~isTI(obj), error('only makes sense for time invariant systems'); end
            
      function [f,df] = myobj(xu)
        f = 0; df = 0*xu;  % feasibility problem, no objective
%         err = [xu-[x0;u0]];
%         f = err'*err;
%         df = 2*err;
      end
      problem.objective = @myobj;
      problem.x0 = [x0;u0];
      
      function [c,ceq,GC,GCeq] = mycon(xu)
        x = xu(1:obj.num_x);
        u = xu(obj.num_x + (1:obj.num_u));

        c=[]; GC=[];
        ceq=[]; GCeq=[];
        
        if (obj.num_xc>0)
          [xdot,df] = geval(@obj.dynamics,0,x,u);
          ceq = [ceq; xdot];
          GCeq = [GCeq, df(:,2:end)'];
        end
        
        if (obj.num_xd>0)
          [xdn,df] = geval(@obj.update,0,x,u);
          ceq=[ceq;x-xdn]; GCeq=[GCeq,([eye(obj.num_x),zeros(obj.num_x,obj.num_u)]-df(:,2:end))'];
        end
        
        if (obj.num_xcon>0)
          [phi,dphi] = geval(@obj.stateConstraints,x);
          ceq = [ceq; phi];
          GCeq = [GCeq, [dphi,zeros(obj.num_xcon,obj.num_u)]'];
        end
      end
      problem.nonlcon = @mycon;
      problem.solver = 'fmincon';

      function stop=drawme(xu,optimValues,state)
        stop=false;
        v.draw(0,xu(1:obj.num_x));
      end
      if (nargin>3 && ~isempty(v))  % useful for debugging (only but only works for URDF manipulators)
        problem.options=optimset('GradObj','on','GradConstr','on','Algorithm','active-set','Display','iter','OutputFcn',@drawme,'TolX',1e-9);
      else
        problem.options=optimset('GradObj','on','GradConstr','on','Algorithm','active-set','Display','off');
      end
      [xu,~,exitflag] = fmincon(problem);
      xstar = Point(obj.getStateFrame,xu(1:obj.num_x));
      ustar = Point(obj.getInputFrame,xu(obj.num_x + (1:obj.num_u)));
      success=(exitflag>0);
      if (~success)
        error('Drake:DrakeSystem:FixedPointSearchFailed','failed to find a fixed point (exitflag=%d)',exitflag);
      end
    end    
  end
  
  % access methods
  methods
    function u = getDefaultInput(obj)
      % Define the default initial input so that behavior is well-defined
      % if no controller is specified or if no control messages have been
      % received yet.
      u = zeros(obj.num_u,1);
    end
    function obj = setNumContStates(obj,num_xc)
      % Guards the num_states variable
      if (num_xc<0), error('num_xc must be >= 0'); end
      obj.num_xc = num_xc;
      obj.num_x = obj.num_xd + obj.num_xc;
      if (isempty(obj.getStateFrame) || obj.num_x~=obj.getStateFrame.dim)
        obj=setStateFrame(obj,CoordinateFrame([class(obj),'State'],obj.num_x,'x'));
      end
    end
    function obj = setNumDiscStates(obj,num_xd)
      % Guards the num_states variable
      if (num_xd<0), error('num_xd must be >= 0'); end
      obj.num_xd = num_xd;
      obj.num_x = obj.num_xc + obj.num_xd;
      if (isempty(obj.getStateFrame) || obj.num_x~=obj.getStateFrame.dim)
        obj=setStateFrame(obj,CoordinateFrame([class(obj),'State'],obj.num_x,'x'));
      end
    end
    function obj = setNumInputs(obj,num_u)
      % Guards the num_u variable.
      %  Also pads umin and umax for any new inputs with [-inf,inf].

      if (num_u<0), error('num_u must be >=0 or DYNAMICALLY_SIZED'); end
      
       % cut umin and umax to the right size, and pad new inputs with
      % [-inf,inf]
      if (length(obj.umin)~=num_u)
        obj.umin = [obj.umin; -inf*ones(max(num_u-length(obj.umin),0),1)];
      end
      if (length(obj.umax)~=num_u)
        obj.umax = [obj.umax; inf*ones(max(num_u-length(obj.umax),0),1)];
      end
      
      obj.num_u = num_u;
      if (isempty(obj.getInputFrame) || obj.num_u~=obj.getInputFrame.dim)
        obj=setInputFrame(obj,CoordinateFrame([class(obj),'Input'],num_u,'u'));
      end
    end
    function obj = setInputLimits(obj,umin,umax)
      % Guards the input limits to make sure it stay consistent
      
      if (isscalar(umin)), umin=repmat(umin,obj.num_u,1); end
      if (isscalar(umax)), umax=repmat(umax,obj.num_u,1); end
      
      sizecheck(umin,[obj.num_u,1]);
      sizecheck(umax,[obj.num_u,1]);
      if (any(obj.umax<obj.umin)), error('umin must be less than umax'); end
      obj.umin = umin;
      obj.umax = umax;
    end
    function obj = setNumOutputs(obj,num_y)
      % Guards the number of outputs to make sure it's consistent
      if (num_y<0), error('num_y must be >=0'); end
      obj.num_y = num_y;
      if (isempty(obj.getOutputFrame) || obj.num_y~=obj.getOutputFrame.dim)
        obj=setOutputFrame(obj,CoordinateFrame([class(obj),'Output'],num_y,'y'));
      end
    end
    function n = getNumZeroCrossings(obj)
      % Returns the number of zero crossings
      n = obj.num_zcs;
    end
    function obj = setNumZeroCrossings(obj,num_zcs)
      % Guards the number of zero crossings to make sure it's valid.
      if (num_zcs<0), error('num_zcs must be >=0'); end
      obj.num_zcs = num_zcs;
    end
    function n = getNumStateConstraints(obj)
      % Returns the number of zero crossings
      n = obj.num_xcon;
    end
    function obj = setNumStateConstraints(obj,num_xcon)
      % Guards the number of zero crossings to make sure it's valid.
      if (num_xcon<0), error('num_xcon must be >=0'); end
      obj.num_xcon = num_xcon;
    end
  end

  % utility methods
  methods
    function [A,B,C,D,x0dot,y0] = linearize(obj,t0,x0,u0)
      % Uses the geval engine to linearize the model around the nominal
      % point, at least for the simple case. 
      
      if (~isCT(obj) || getNumDiscStates(obj)>0)  % boot if it's not the simple case
        [A,B,C,D,x0dot,y0] = linearize@DynamicalSystem(obj,t0,x0,u0);
        return;
      end
      
      nX = getNumContStates(obj);
      nU = getNumInputs(obj);
      [~,df] = geval(@obj.dynamics,t0,x0,u0);
      A = df(:,1+(1:nX));
      B = df(:,nX+1+(1:nU));
      
      if (nargout>2)
        [~,dy] = geval(@obj.output,t0,x0,u0);
        C = dy(:,1+(1:nX));
        D = dy(:,nX+1+(1:nU));
        if (nargout>4)
          x0dot = dynamics(obj,t0,x0,u0);
          if (nargout>5)
            y0 = output(obj,t0,x0,u0);
          end
        end
      end
    end

    function traj = simulateODE(obj,tspan,x0,options)
      % Simulates the system using the ODE45 suite of solvers
      % instead of the simulink solvers.  
      %
      % @param tspan a 1x2 vector of the form [t0 tf]
      % @param x0 a vector of length(getNumStates) which contains the initial
      % state
      % @param options options structure
      %
      % No options implemented yet

      if (nargin<3), x0=getInitialState(obj); end
      
      if (obj.num_zcs>0), warning('Drake:DrakeSystem:UnsupportedZeroCrossings','system has zero-crossings, but i havne''t passed them to ode45 yet.  (should be trivial)'); end
      if (obj.num_xcon>0), warning('Drake:DrakeSystem:UnsupportedConstraints','system has constraints, but they are not explicitly satisfied during simulation (yet - it should be an easy fix in the ode suite)'); end
      
      odeoptions = obj.simulink_params;
      odefun = @(t,x)obj.dynamics(t,x,zeros(obj.getNumInputs(),1));
      if (isfield(obj.simulink_params,'Solver'))
        sol = feval(obj.simulink_params.Solver,odefun,tspan,x0,odeoptions);
      else
        sol = ode45(odefun,tspan,x0,odeoptions);
      end
      xtraj = ODESolTrajectory(sol);
      traj = FunctionHandleTrajectory(@(t)obj.output(t,xtraj.eval(t),zeros(obj.getNumInputs(),1)),[obj.getNumOutputs,1],tspan);
    end
    
    function sys=feedback(sys1,sys2)
      % Constructs a feedback combination of sys1 and sys2.  
      %
      % @param sys1 first DynamicalSystem (on the forward path)
      % @param sys2 second DynamicalSystem (on the backward path)
      %
      % The input to the feedback model is added to the output of sys2
      % before becoming the input for sys1.  The output of the feedback
      % model is the output of sys1.

      if isa(sys2,'DrakeSystem')
        try 
          sys=FeedbackSystem(sys1,sys2);  % try to keep it a drakesystem
        catch ex
          if (strcmp(ex.identifier, 'Drake:DrakeSystem:UnsupportedSampleTime'))
            warning('Drake:DrakeSystem:UnsupportedSampleTime','Aborting feedback combination as a DrakeSystem due to incompatible sample times');
            sys = feedback@DynamicalSystem(sys1,sys2);
          elseif (strcmp(ex.identifier, 'Drake:FeedbackSystem:NoHybridSupport') || strcmp(ex.identifier,'Drake:FeedbackSystem:NoStochasticSupport'))
            sys = feedback@DynamicalSystem(sys1,sys2);
          else
            rethrow(ex);
          end
        end
      else
        sys=feedback@DynamicalSystem(sys1,sys2);
      end
    end
    
    function sys=cascade(sys1,sys2)
      % Constructs a cascade combination of sys1 and sys2.  
      %
      % @param sys1 first DynamicalSystem 
      % @param sys2 second DynamicalSystem 
      %
      % The input to the cascade system is the input to sys1.  
      % The output of sys1 is fed to the input of sys2.
      % The output of the cascade system is the output of sys2.

      if isa(sys2,'DrakeSystem')
        try
          sys=CascadeSystem(sys1,sys2);   % try to keep it a drakesystem
        catch ex
          if (strcmp(ex.identifier, 'Drake:DrakeSystem:UnsupportedSampleTime'))
            warning('Drake:DrakeSystem:UnsupportedSampleTime','Aborting cascade combination as a DrakeSystem due to incompatible sample times');
            sys = cascade@DynamicalSystem(sys1,sys2);
          elseif (strcmp(ex.identifier, 'Drake:CascadeSystem:NoHybridSupport') || strcmp(ex.identifier,'Drake:CascadeSystem:NoStochasticSupport'))
            sys = cascade@DynamicalSystem(sys1,sys2);
          else
            rethrow(ex);
          end
        end
      else
        sys=cascade@DynamicalSystem(sys1,sys2);
      end
    end
    
    function polysys = extractPolynomialSystem(obj)
      % Attempts to symbolically extract the extra structure of a
      % polynomial system from the Drake system
      % Will throw an error if the system is not truly polynomial.
      %
      % See also extractTrigPolySystem, taylorApprox
      
      t=msspoly('t',1);
      x=msspoly('x',sys.num_x);
      u=msspoly('u',sys.num_u);
      
      p_dynamics_rhs=[];
      p_dynamics_lhs=[];
      p_update = [];
      p_output = [];
      p_state_constraints = [];
      
      try 
        if (obj.num_xc>0)
          p_dynamics_rhs = dynamics(obj,t,x,u);
        end
        if (obj.num_xd>0)
          p_update = update(obj,t,x,u);
        end
        p_output = output(obj,t,x,u);
        
        if (obj.num_xcon>0)
          p_state_constraints = stateConstraints(obj,x);
        end
      catch ex
        error('DrakeSystem:ExtractPolynomialSystem:NotPolynomial','This system appears to not be polynomial');
      end
      polysys = SpotPolynomialSystem(getInputFrame(obj),getStateFrame(obj),getOutputFrame(obj),p_dynamics_rhs,p_dynamics_lhs,p_update,p_output,p_state_constraints);
      
      polysys = setSampleTime(polysys,obj.getSampleTime);
    end
    
    function sys = extractAffineSystem(obj)
      % Attempts to symbolically extract the extra structure of an
      % affine system from the Drake system
      % Will throw an error if the system is not truly affine.
      %
      % See also taylorApprox
      
      sys = extractAffineSystem(extractPolynomialSystem(obj));
    end

    function sys = extractLinearSystem(obj)
      % Attempts to symbolically extract the extra structure of a
      % linear system from the Drake system
      % Will throw an error if the system is not truly linear.
      %
      % See also linearize, taylorApprox

      sys = extractLinearSystem(extractAffineSystem(obj));
    end
  end
  
  methods % deprecated (due to refactoring)
    function polysys = makeTrigPolySystem(obj,options)
      % deprecated method (due to refactoring): please use extractTrigPolySystem instead
      warning('Drake:DeprecatedMethod','makeTrigPolySystem has been refactored and will go away.  Please use extractTrigPolySystem(obj,options) instead');
      polysys = extractTrigPolySystem(obj,options);
    end
  end
  
  % utility methods
  methods
    function gradTest(obj,t,x,u,options)
      % Compare numerical and analytical derivatives of dynamics,update,and
      % output
      
      if nargin<2, t=0; end
      if nargin<3, x=getInitialState(obj); end
      if nargin<4, u=getDefaultInput(obj); end
      if nargin<5, options=struct('tol',.01); end
      if ~isfield(options,'dynamics'), options.dynamics=true; end
      if ~isfield(options,'update'), options.update=true; end
      if ~isfield(options,'output'), options.output=true; end
      
      if (options.dynamics && getNumContStates(obj))
        gradTest(@obj.dynamics,t,x,u,options)
      end
      if (options.update && getNumDiscStates(obj))
        gradTest(@obj.update,t,x,u,options);
      end
      if (options.output && getNumOutputs(obj))
        gradTest(@obj.output,t,x,u,options);
      end
    end
  end
  
  properties (SetAccess=private, GetAccess=protected)
    num_xc=0; % number of continuous state variables
    num_xd=0; % number of dicrete(-time) state variables
    num_x=0;  % dimension of x (= num_xc + num_xd)
    num_u=0;  % dimension of u
    num_y=0;  % dimension of the output y
    num_zcs = 0;  % number of zero-crossings.  @default: 0
    uid;    % unique identifier for simulink models of this block instance
    direct_feedthrough_flag=true;  % true/false: does the output depend on u?  set false if you can!
    ts=[];    % default sample times of the model
  end
  properties (SetAccess=protected, GetAccess=protected)
    num_xcon = 0; % number of state constraints. @default: 0
  end
  properties (SetAccess=private, GetAccess=public)
    umin=[];   % constrains u>=umin (default umin=-inf)
    umax=[];    % constrains u<=uman (default umax=inf)
  end
  
end
